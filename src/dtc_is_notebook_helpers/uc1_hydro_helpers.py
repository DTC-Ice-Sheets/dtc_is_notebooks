"""
Use Case 1 - Hydropower. The machinery behind the demonstrator notebook.

The method follows David Parkes' original demonstrator: a single number
describing a melt season is used to predict discharge in each sub-period of
that season separately. One regression per week, twenty-two of them, each
across years. The seasonal number sets how big the year is; the per-week
regressions set the shape.

What has changed from the original
----------------------------------
* twenty seasons of training data rather than eight
* the step length is a parameter, so weekly, fortnightly or daily all work
* annual totals and peak flows come out of the same profile
* the predictor can be the FTS coefficient or the plain catchment total,
  chosen at run time, so the two can be compared on the same footing
* future scenarios can be driven by extracted 2050 projections rather than
  only by perturbing the observed record

Everything is indexed by date and returns pandas objects, so any one step can
be replaced without disturbing those either side of it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# One predictor per season, or several: a named series or a (year x predictor) frame.
Predictors = pd.Series | pd.DataFrame
# One season's predictor values, as a scalar or a row.
Values = float | list[float] | np.ndarray | pd.Series
# A list of seasons.
Years = Sequence[int] | np.ndarray | pd.Index

MELT_MONTHS = (5, 6, 7, 8, 9)
MELT_START_DOY = 121          # 1 May
MELT_END_DOY = 274            # 30 September
MISSING = -9999.0
SECONDS_PER_DAY = 86400.0
M3S_TO_KM3_PER_DAY = SECONDS_PER_DAY / 1e9


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_discharge(source: str | Path, column: int | None = None) -> pd.Series:
    """Daily Watson discharge in m3/s, indexed by date.

    Accepts either the DTC bucket CSV that the original notebook reads
    (`WatsonDischarge2006_2025.csv`, gap-filled series in column 7) or the raw
    tab-separated GEUS Dataverse file. The gap-filled series is the one to use:
    the measurement-only column looks similar and has holes in it.
    """
    text = str(source)
    if text.endswith(".txt"):
        return _load_geus(Path(source))

    frame = pd.read_csv(text)
    if column is None:
        column = 7
    values = pd.to_numeric(frame.iloc[:, column], errors="coerce")

    lower = {str(c).strip().lower(): c for c in frame.columns}
    if {"year", "month", "day"} <= set(lower):
        index = pd.to_datetime({
            "year": frame[lower["year"]].astype(int),
            "month": frame[lower["month"]].astype(int),
            "day": frame[lower["day"]].astype(int)})
    else:
        # No date columns: the file is one row per day from 1 January 2006.
        index = pd.date_range("2006-01-01", periods=len(values), freq="D")

    series = pd.Series(values.to_numpy(), index=index, name="discharge_m3s")
    return series.mask(series == MISSING).sort_index()


def _load_geus(path: Path) -> pd.Series:
    """Read the raw GEUS Dataverse export, tab separated with a named header."""
    header = path.read_text(errors="ignore").split("\n", 1)[0].split("\t")
    columns = {name.strip().lower(): i for i, name in enumerate(header)}
    year_col = columns["year"]
    month_col = next(i for n, i in columns.items() if "month" in n)
    day_col = next(i for n, i in columns.items() if n.strip() in ("day", "dayofmonth"))
    flux_col = next(i for n, i in columns.items()
                    if "divers" in n and "temperature" in n)

    table = np.genfromtxt(path, delimiter="\t", skip_header=1,
                          usecols=(year_col, month_col, day_col, flux_col))
    year, month, day, flux = table.T
    valid = np.isfinite(flux) & (flux != MISSING)
    index = pd.to_datetime({"year": year[valid].astype(int),
                            "month": month[valid].astype(int),
                            "day": day[valid].astype(int)})
    return pd.Series(flux[valid], index=index, name="discharge_m3s").sort_index()


def load_grf_matrix(source: str | Path, start_year: int,
                    log10: bool = True) -> pd.DataFrame:
    """`run_grf_batch.py` output as a (time x mesh node) frame.

    The matrix is mesh nodes by monthly snapshots with no dates attached, so
    the caller supplies the first melt season. Five months a season, May to
    September.
    """
    matrix = np.loadtxt(source, delimiter=",")
    values = np.power(10.0, matrix) if log10 else matrix
    n_seasons = values.shape[1] // len(MELT_MONTHS)
    stamps = [pd.Timestamp(year=start_year + s, month=m, day=1)
              for s in range(n_seasons) for m in MELT_MONTHS]
    return pd.DataFrame(values[:, :len(stamps)].T, index=pd.DatetimeIndex(stamps))


def load_seasonal_eo(source: str | Path, years: Years) -> pd.Series:
    """CryoSat-2 seasonal elevation change, catchment mean, one value a season.

    The matrix is mesh nodes by seasons, already harmonised by GRF. Values are
    metres of surface lowering over the melt season, so they are negative in
    almost every year and are never log-transformed.
    """
    matrix = np.loadtxt(source, delimiter=",")
    return pd.Series(np.nanmean(matrix, axis=0), index=list(years),
                     name="elevation_change")


def load_coefficients(source: str | Path) -> pd.DataFrame:
    """FTS coefficients as written by `model.coeff()` - one row per season.

    The cache is written with a row index, so read it back as one. Without
    this the index column is taken as data and every factor shifts one place
    right, which is silent and changes the answer: a cached run then has the
    row numbers 0, 1, 2 ... in place of the first factor.
    """
    frame = pd.read_csv(source)
    if str(frame.columns[0]).startswith("Unnamed"):
        frame = frame.set_index(frame.columns[0])
        frame.index.name = None
    frame.columns = [f"factor_{i + 1}" for i in range(frame.shape[1])]
    return frame


# --------------------------------------------------------------------------
# From mesh fields to one number a season
# --------------------------------------------------------------------------

def catchment_mean(fields: pd.DataFrame, weights: np.ndarray | None = None) -> pd.Series:
    """Collapse a (time x node) frame to one value per time step."""
    if weights is None:
        series = fields.mean(axis=1)
    else:
        w = np.asarray(weights, dtype=float)
        series = pd.Series(fields.to_numpy() @ (w / w.sum()), index=fields.index)
    return series.rename("runoff")


def seasonal_predictor(runoff: pd.Series) -> pd.Series:
    """Melt-season runoff total, one number per year.

    This is the alternative to the FTS coefficient, and on this catchment it
    predicts discharge rather better. Both are accepted by `ProfileModel`, so
    the notebook can show them side by side.
    """
    melt = runoff[runoff.index.month.isin(MELT_MONTHS)]
    counts = melt.groupby(melt.index.year).size()
    # A season the archive only half covers sums to a small total and would be
    # read as a quiet melt year. RACMO stops on 31 August 2021, so this matters.
    complete = counts[counts == counts.max()].index
    totals = melt.groupby(melt.index.year).sum()
    dropped = [int(y) for y in totals.index if y not in complete]
    if dropped:
        logger.info("dropping incomplete melt season(s): %s", dropped)
    return totals.loc[complete].rename("seasonal_runoff")


# --------------------------------------------------------------------------
# The model: one regression per sub-period of the melt season
# --------------------------------------------------------------------------

@dataclass
class ProfileModel:
    """Predicts a whole within-season discharge profile from seasonal numbers.

    For each step of the melt season - week 1, week 2, and so on - a linear
    model is fitted across years between the seasonal predictor(s) and the
    discharge measured in that step. Twenty-two weeks gives twenty-two models.
    Feeding one season's predictor values through all of them at once returns a
    complete profile, so the result has a shape and a peak rather than just a
    total.

    This is the structure of the original demonstrator, with the step length
    made a parameter, the training set extended to the full gauged record, and
    room for more than one predictor so that an EO dataset can be added
    alongside the climate model.
    """

    starts: np.ndarray            # day-of-year at the start of each step
    step_days: int
    coefficients: np.ndarray      # steps x (1 + n predictors), intercept first
    residual_sd: np.ndarray
    r2: np.ndarray
    years: np.ndarray
    names: list
    # (X'X)^-1 and the residual degrees of freedom, per step. Needed for a
    # genuine prediction interval: without them the band is only the scatter of
    # the fit, and takes no account of how far the predictor sits from the data
    # the model was fitted on.
    xtx_inv: np.ndarray | None = None
    dof: np.ndarray | None = None

    # ---------------------------------------------------------------- fit --
    @classmethod
    def fit(cls, discharge: pd.Series, predictors: Predictors,
            step_days: int = 7) -> "ProfileModel":
        """
        Fit one regression per step of the melt season, across seasons.

        Parameters
        ----------
        discharge : pandas.Series
            Daily gauge discharge, indexed by date.
        predictors : pandas.Series or pandas.DataFrame
            One value per season, or several, indexed by year.
        step_days : int, optional
            Length of each sub-period in days.

        Returns
        -------
        ProfileModel
            The fitted model.
        """
        table = _as_frame(predictors)
        steps = step_matrix(discharge, step_days)
        years = np.array([y for y in steps.index if y in table.index])
        steps = steps.loc[years]
        design = np.column_stack([np.ones(len(years)), table.loc[years].to_numpy(float)])

        n_steps = steps.shape[1]
        n_terms = design.shape[1]
        coefficients = np.full((n_steps, n_terms), np.nan)
        residual_sd = np.full(n_steps, np.nan)
        r2 = np.full(n_steps, np.nan)
        xtx_inv = np.full((n_steps, n_terms, n_terms), np.nan)
        dof_per_step = np.full(n_steps, np.nan)

        for j in range(n_steps):
            y = steps.iloc[:, j].to_numpy(dtype=float)
            keep = np.isfinite(y) & np.isfinite(design).all(axis=1)
            if keep.sum() < design.shape[1] + 1:
                continue
            beta, *_ = np.linalg.lstsq(design[keep], y[keep], rcond=None)
            residuals = y[keep] - design[keep] @ beta
            dof = max(keep.sum() - design.shape[1], 1)
            coefficients[j] = beta
            residual_sd[j] = float(np.sqrt((residuals ** 2).sum() / dof))
            dof_per_step[j] = dof
            try:
                xtx_inv[j] = np.linalg.inv(design[keep].T @ design[keep])
            except np.linalg.LinAlgError:
                pass
            denominator = ((y[keep] - y[keep].mean()) ** 2).sum()
            r2[j] = float(1 - (residuals ** 2).sum() / denominator) if denominator else np.nan

        return cls(np.asarray(steps.columns, dtype=int), step_days, coefficients,
                   residual_sd, r2, years, list(table.columns),
                   xtx_inv=xtx_inv, dof=dof_per_step)

    # ------------------------------------------------------------ predict --
    def profile(self, values: Values) -> pd.Series:
        """Discharge through one melt season, m3/s, for one season's predictors."""
        row = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
        vector = np.concatenate([[1.0], row])
        return pd.Series(self.coefficients @ vector,
                         index=self.starts, name="m3s").clip(lower=0)

    def profiles(self, predictors: Predictors) -> pd.DataFrame:
        """One profile per year: rows are years, columns day-of-year."""
        table = _as_frame(predictors)
        return pd.DataFrame({year: self.profile(row.to_numpy())
                             for year, row in table.iterrows()}).T

    def prediction_se(self, values: Values) -> np.ndarray:
        """Give the standard error of a *new* observation, per step.

        A prediction interval is wider than the residual scatter for two
        reasons: the fitted line is itself uncertain, and that uncertainty
        grows with distance from the predictor values the model was fitted on.
        Both are carried by the leverage term below, so the band widens on its
        own when a scenario sits outside the historical range - which is the
        behaviour you want when predicting 2050 from a model trained on
        2006-2020.
        """
        row = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
        vector = np.concatenate([[1.0], row])
        if self.xtx_inv is None:
            return self.residual_sd.copy()
        leverage = np.einsum("i,kij,j->k", vector, self.xtx_inv, vector)
        return self.residual_sd * np.sqrt(1.0 + np.clip(leverage, 0.0, None))

    def leverage(self, values: Values) -> float:
        """How far outside the fitted range a predictor sits, as a ratio.

        1.0 means the band is as wide as it would be at the centre of the
        training data. Larger values mean extrapolation; the number is the
        factor by which the interval has widened in response.
        """
        if self.xtx_inv is None:
            return 1.0
        centre = np.nanmean(self.prediction_se(np.nanmean(_design_rows(self), axis=0)))
        here = np.nanmean(self.prediction_se(values))
        return float(here / centre) if centre else 1.0

    def band(self, values: Values, sigmas: float | None = None,
             level: float = 0.68) -> pd.DataFrame:
        """Prediction interval around one profile.

        Pass `level` as the coverage you want - 0.68 for one sigma, 0.95 for
        the interval infrastructure design usually asks for. The multiplier is
        Student's t on the fitted degrees of freedom, not the normal
        quantile: with fifteen seasons and two terms there are thirteen
        degrees of freedom, where t is 2.16 against the normal 1.96, so
        assuming normality would understate a 95% band by about ten per cent.

        `sigmas` is kept for backwards compatibility and overrides `level`.
        """
        centre = self.profile(values)
        spread = pd.Series(self.prediction_se(values) * self._multiplier(sigmas, level),
                           index=self.starts)
        return pd.DataFrame({"lower": (centre - spread).clip(lower=0),
                             "predicted": centre,
                             "upper": centre + spread})

    def bands(self, values: Values,
              levels: Sequence[float] = (0.68, 0.95)) -> pd.DataFrame:
        """Several coverages at once: columns lower_68, upper_68, lower_95, ..."""
        out = {"predicted": self.profile(values)}
        for level in levels:
            b = self.band(values, level=level)
            tag = f"{round(level * 100)}"
            out[f"lower_{tag}"], out[f"upper_{tag}"] = b["lower"], b["upper"]
        return pd.DataFrame(out)

    def _multiplier(self, sigmas: float | None, level: float) -> np.ndarray:
        """T multiplier per step, or the plain sigma count if one was given."""
        if sigmas is not None:
            return np.full(len(self.starts), float(sigmas))
        dof = (self.dof if self.dof is not None
               else np.full(len(self.starts), max(len(self.years) - 2, 1)))
        return np.array([_t_quantile(0.5 + level / 2.0, max(int(d), 1))
                         if np.isfinite(d) else np.nan for d in dof])

    def __str__(self) -> str:
        """Summarise the fit in one line."""
        return (f"ProfileModel: {len(self.starts)} steps of {self.step_days} days, "
                f"{len(self.years)} seasons ({self.years.min()}-{self.years.max()}), "
                f"predictors {', '.join(self.names)}, "
                f"median step r2 {np.nanmedian(self.r2):.2f}")


def _design_rows(model: "ProfileModel") -> np.ndarray:
    """Recover the predictor values the model was fitted on, from (X'X)^-1.

    Only the centroid is needed, and for a fitted design that is available from
    the inverse cross-product: the intercept row of (X'X)^-1 encodes it.
    """
    inv = model.xtx_inv[np.isfinite(model.xtx_inv).all(axis=(1, 2))]
    if not len(inv):
        return np.zeros((1, max(len(model.names), 1)))
    xtx = np.linalg.inv(inv[0])
    n = xtx[0, 0]
    return (xtx[0, 1:] / n)[None, :] if n else np.zeros((1, xtx.shape[0] - 1))


def _t_quantile(p: float, dof: int) -> float:
    """Two-sided Student's t quantile, without a scipy dependency."""
    try:
        from scipy.stats import t
        return float(t.ppf(p, dof))
    except ImportError:
        pass
    # Cornish-Fisher expansion on the normal quantile; better than 0.5% for
    # dof >= 5, which is the regime the melt-season regressions sit in.
    z = _normal_quantile(p)
    g1 = (z ** 3 + z) / 4.0
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96.0
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384.0
    return z + g1 / dof + g2 / dof ** 2 + g3 / dof ** 3


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _as_frame(predictors: Predictors) -> pd.DataFrame:
    """One predictor or several, always as a (year x predictor) frame."""
    if isinstance(predictors, pd.Series):
        return predictors.to_frame(predictors.name or "predictor").dropna()
    return pd.DataFrame(predictors).dropna()


def step_matrix(discharge: pd.Series, step_days: int = 7) -> pd.DataFrame:
    """Mean discharge in each step of each melt season.

    Rows are years, columns the day-of-year each step begins. Steps are cut on
    day-of-year rather than calendar weeks so that every year lines up, which
    matters for fitting one regression per step across years.
    """
    starts = np.arange(MELT_START_DOY, MELT_END_DOY - step_days + 2, step_days)
    doy = discharge.index.dayofyear
    year = discharge.index.year
    rows = {}
    for y in np.unique(year):
        in_year = year == y
        values = []
        for start in starts:
            window = in_year & (doy >= start) & (doy < start + step_days)
            chunk = discharge[window]
            values.append(float(chunk.mean()) if len(chunk) else np.nan)
        rows[int(y)] = values
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=starts)
    return frame.dropna(how="all")


def leave_one_year_out(discharge: pd.Series, predictors: Predictors,
                       step_days: int = 7) -> pd.DataFrame:
    """Predict each season's profile from a model that never saw that season."""
    table = _as_frame(predictors)
    observed = step_matrix(discharge, step_days)
    years = np.array([y for y in observed.index if y in table.index])
    predicted = pd.DataFrame(index=years, columns=observed.columns, dtype=float)
    for year in years:
        model = ProfileModel.fit(discharge[discharge.index.year != year],
                                 table.drop(index=year), step_days)
        predicted.loc[year] = model.profile(table.loc[year].to_numpy()).to_numpy()
    return pd.concat({"observed": observed.loc[years], "predicted": predicted}, axis=1)


def profile_skill(validation: pd.DataFrame) -> dict:
    """r2 and RMSE over every step of every held-out season."""
    observed = validation["observed"].to_numpy(dtype=float).ravel()
    predicted = validation["predicted"].to_numpy(dtype=float).ravel()
    keep = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[keep], predicted[keep]
    ss_res = ((observed - predicted) ** 2).sum()
    ss_tot = ((observed - observed.mean()) ** 2).sum()
    return {"n_steps": int(keep.sum()),
            "r2": float(1 - ss_res / ss_tot),
            "rmse_m3s": float(np.sqrt(ss_res / keep.sum()))}


# --------------------------------------------------------------------------
# Stakeholder numbers
# --------------------------------------------------------------------------

def annual_total(profile: pd.Series, step_days: int) -> float:
    """Melt-season volume in km3 from a profile of mean flows."""
    return float(profile.sum() * step_days * M3S_TO_KM3_PER_DAY)


def annual_peak(profile: pd.Series) -> tuple[float, int]:
    """Largest step in a profile, and the day of year it starts."""
    idx = int(np.nanargmax(profile.to_numpy()))
    return float(profile.iloc[idx]), int(profile.index[idx])


def summarise(profiles: pd.DataFrame, step_days: int) -> pd.DataFrame:
    """Total, peak and peak timing for every year in a set of profiles."""
    rows = []
    for year, profile in profiles.iterrows():
        peak, start_doy = annual_peak(profile)
        date = pd.Timestamp(year=int(year), month=1, day=1) + pd.Timedelta(days=int(start_doy) - 1)
        rows.append({"year": int(year),
                     "total_km3": annual_total(profile, step_days),
                     "peak_m3s": peak,
                     "peak_week_starting": date.strftime("%d %b")})
    return pd.DataFrame(rows).set_index("year")


def steps_above(profiles: pd.DataFrame, threshold_m3s: float) -> pd.Series:
    """Count the steps per year above an operating threshold."""
    return (profiles > threshold_m3s).sum(axis=1).rename("steps_above")


# --------------------------------------------------------------------------
# Future scenarios
# --------------------------------------------------------------------------

@dataclass
class Scenario:
    """A future seasonal-predictor distribution to run the model over.

    Two ways to build one:

    `from_projection` - the real case. Point it at an extracted 2050 MAR or
    RACMO run, processed exactly like the historical forcing, and it returns
    the seasonal predictor values those years imply. The model, fitted on the
    observed period, is then applied forward.

    `from_perturbation` - the stand-in until those runs are reachable. It
    reshapes the observed predictor distribution: `weight` slides the centre
    between the driest and wettest observed seasons, `variability` widens or
    narrows the spread. This mirrors the controls in the original demonstrator.
    """

    name: str
    values: pd.Series
    notes: str = field(default="")

    @classmethod
    def from_projection(cls, name: str, source: str | Path, start_year: int,
                        notes: str = "", **kwargs: object) -> "Scenario":
        """
        Build a scenario from an extracted projection on disk.

        Parameters
        ----------
        name : str
            Label for the scenario.
        source : str or pathlib.Path
            The fitted matrix to read.
        start_year : int
            First season in the matrix.
        notes : str, optional
            A line describing where the scenario came from.
        **kwargs : object
            Passed through to :func:`load_grf_matrix`.

        Returns
        -------
        Scenario
            The scenario, ready to run.
        """
        fields = load_grf_matrix(source, start_year, **kwargs)
        return cls(name, seasonal_predictor(catchment_mean(fields)),
                   notes or f"extracted projection, {source}")

    @classmethod
    def from_perturbation(cls, name: str, observed: pd.Series, weight: float = 0.5,
                          variability: float = 1.0, n_years: int = 20,
                          seed: int = 0, notes: str = "") -> "Scenario":
        """
        Build a scenario by perturbing the observed seasonal record.

        Parameters
        ----------
        name : str
            Label for the scenario.
        observed : pandas.Series
            The observed seasonal predictor.
        weight : float, optional
            Where to sit between the observed minimum and maximum; 0.5 is the mean.
        variability : float, optional
            Scale on the year-to-year spread.
        n_years : int, optional
            How many seasons to generate.
        seed : int, optional
            Seed for the generator, so a run is repeatable.
        notes : str, optional
            A line describing the scenario.

        Returns
        -------
        Scenario
            The scenario, ready to run.
        """
        rng = np.random.default_rng(seed)
        centre = observed.mean() + (weight - 0.5) * 2 * observed.std()
        draws = rng.normal(centre, observed.std() * variability, n_years)
        index = pd.RangeIndex(2041, 2041 + n_years, name="year")
        return cls(name, pd.Series(draws, index=index),
                   notes or f"weight {weight:g}, variability {variability:g}")

    def run(self, model: ProfileModel) -> pd.DataFrame:
        """Every year of the scenario as a profile.

        A model fitted with extra predictors needs values for them too; the
        scenario carries only the climate-model term, so the others are set to
        zero. A future EO dataset does not exist to be perturbed, and zero is
        the value that leaves the climate term carrying the scenario alone.
        """
        table = _as_frame(self.values)
        if len(model.names) > table.shape[1]:
            for name in model.names[table.shape[1]:]:
                table[name] = 0.0
        return model.profiles(table)

    def __str__(self) -> str:
        """Summarise the scenario in one line."""
        return f"{self.name}: {len(self.values)} seasons, {self.notes}"


def compare(scenarios: dict[str, pd.DataFrame], step_days: int) -> pd.DataFrame:
    """One row per scenario: mean, range and peak of the annual totals."""
    rows = []
    for name, profiles in scenarios.items():
        table = summarise(profiles, step_days)
        rows.append({"scenario": name,
                     "mean total (km3)": table["total_km3"].mean(),
                     "driest (km3)": table["total_km3"].min(),
                     "wettest (km3)": table["total_km3"].max(),
                     "mean peak (m3/s)": table["peak_m3s"].mean(),
                     "highest peak (m3/s)": table["peak_m3s"].max()})
    return pd.DataFrame(rows).set_index("scenario")


# --------------------------------------------------------------------------
# Uncertainty propagation
# --------------------------------------------------------------------------

MEAN_SD_COLUMN = "__mean_sd__"


def catchment_sd(sd_fields: pd.DataFrame, correlated: bool = True) -> pd.Series:
    """Spread of the catchment mean implied by the GRF posterior, per snapshot.

    The module publishes this directly as `v_uncertainty_mean`, and `run_grf`
    carries it through as a named column. Where that is present it is used - it
    comes from the joint posterior, which is the correct basis.

    Where it is absent, the node-wise spreads are collapsed instead. That is an
    approximation with two far-apart limits, because a fitted field is smooth
    and neighbouring nodes err together:

        independent   sd(mean) = sqrt(sum(sd^2)) / n      - optimistic
        correlated    sd(mean) = mean(sd)                 - conservative

    `correlated=True` takes the conservative one, which is the right default
    for a number that will be quoted.
    """
    if MEAN_SD_COLUMN in sd_fields.columns:
        return sd_fields[MEAN_SD_COLUMN].astype(float).rename("catchment_sd")

    values = sd_fields.to_numpy(dtype=float)
    if correlated:
        spread = np.nanmean(values, axis=1)
    else:
        spread = np.sqrt(np.nansum(values ** 2, axis=1)) / values.shape[1]
    return pd.Series(spread, index=sd_fields.index, name="catchment_sd")


def seasonal_sd(snapshot_sd: pd.Series, years: Years, monthly_values: pd.Series | None = None,
                log10: bool = True, draws: int = 4000, seed: int = 0) -> pd.Series:
    """Spread on the seasonal predictor, in the units the predictor is in.

    The GRF fits log10 runoff, so its posterior spread is in log10 units, while
    the seasonal predictor is a sum of *back-transformed* monthly catchment
    means. A log-space spread therefore acts multiplicatively, not additively:
    an uncertainty of s on log10(x) means x is uncertain by a factor of 10**s.

    So this resamples rather than adding in quadrature. For each month a
    log-offset is drawn, the month is scaled by 10**offset, and the months are
    summed. Months are drawn independently, which is right: each is a separate
    INLA fit on separate observations.

    Pass `log10=False` for a predictor already in linear units, and the months
    combine in quadrature as before.
    """
    frame = pd.DataFrame({"sd": snapshot_sd.to_numpy(dtype=float)}, index=list(years))

    if not log10 or monthly_values is None:
        grouped = frame.groupby(level=0)["sd"]
        return np.sqrt(grouped.apply(lambda s: (s ** 2).sum())).rename("seasonal_sd")

    frame["value"] = np.asarray(monthly_values, dtype=float)[:len(frame)]
    rng = np.random.default_rng(seed)
    spread = {}
    for year, block in frame.groupby(level=0):
        base = block["value"].to_numpy()
        sigma = block["sd"].to_numpy()
        offsets = rng.normal(0.0, 1.0, size=(draws, len(base))) * sigma[None, :]
        totals = (base[None, :] * np.power(10.0, offsets)).sum(axis=1)
        spread[year] = float(totals.std())
    return pd.Series(spread, name="seasonal_sd")


def multiplicative_factor(snapshot_sd: pd.Series) -> float:
    """Give the factor a log10 spread implies, for a sanity check.

    A spread of 0.3 in log10 is a factor of 2; 0.7 is a factor of 5. If this
    comes back large, the quantity the module is reporting is probably not a
    posterior standard deviation of the fitted field, and it is worth asking
    before the number is propagated into anything anyone relies on.
    """
    return float(10.0 ** np.nanmean(snapshot_sd.to_numpy(dtype=float)))


def propagate(model: "ProfileModel", value: Values, value_sd: float = 0.0,
              draws: int = 2000, sigmas: float | None = None,
              level: float = 0.68, seed: int = 0) -> pd.DataFrame:
    """Prediction band including harmonisation uncertainty, by Monte Carlo.

    Two sources, combined properly rather than added:

    * the seasonal predictor is uncertain because the GRF fit is uncertain, so
      it is resampled from N(value, value_sd);
    * each weekly relationship has residual scatter, so a residual is drawn for
      every step of every draw.

    With `value_sd = 0` this reduces to the regression prediction interval,
    which is what the notebook shows when the module returns no posterior
    spread. The difference between the two bands is exactly the contribution
    harmonisation makes.
    """
    rng = np.random.default_rng(seed)
    row = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    perturbed = row[None, :] + rng.normal(0.0, value_sd, size=(draws, len(row)))
    design = np.column_stack([np.ones(draws), perturbed])
    centres = design @ model.coefficients.T                     # draws x steps
    # prediction_se, not residual_sd: includes the uncertainty of the fitted
    # line and widens where the predictor sits away from the fitted range.
    spread = model.prediction_se(row)
    noise = rng.normal(0.0, 1.0, size=centres.shape) * spread[None, :]
    sampled = np.clip(centres + noise, 0.0, None)

    lower = (50 * (1 - _coverage(sigmas)) if sigmas is not None
             else 50 * (1 - level))
    return pd.DataFrame({"lower": np.percentile(sampled, lower, axis=0),
                         "predicted": sampled.mean(axis=0),
                         "upper": np.percentile(sampled, 100 - lower, axis=0)},
                        index=model.starts)


def _coverage(sigmas: float) -> float:
    """Two-sided normal coverage for a number of standard deviations."""
    from math import erf, sqrt
    return erf(sigmas / sqrt(2.0))


def uncertainty_budget(model: "ProfileModel", value: Values, value_sd: float,
                       draws: int = 2000) -> pd.DataFrame:
    """How wide the band is with and without the harmonisation term."""
    fitting = propagate(model, value, 0.0, draws=draws)
    both = propagate(model, value, value_sd, draws=draws)
    width_fitting = (fitting["upper"] - fitting["lower"]).mean()
    width_both = (both["upper"] - both["lower"]).mean()
    return pd.DataFrame({
        "mean band width (m3/s)": [width_fitting, width_both],
        "share from harmonisation": [0.0, 1 - (width_fitting / width_both) ** 2],
    }, index=["fitting only", "fitting + harmonisation"])


# --------------------------------------------------------------------------
# Is the model worth having? Two reference points
# --------------------------------------------------------------------------

# The Watson catchment, from the outline the mesh is built on. Only used to
# turn catchment-mean runoff in mm into a flow, so that runoff and discharge
# can be put on the same axis.
CATCHMENT_AREA_M2 = 5.746e9


def monthly_flow(monthly_mm: pd.Series,
                 area_m2: float = CATCHMENT_AREA_M2) -> pd.Series:
    """Catchment-mean runoff in mm per month, as a mean flow in m3/s."""
    days = monthly_mm.index.days_in_month.to_numpy(dtype=float)
    return pd.Series(monthly_mm.to_numpy(dtype=float) / 1000.0 * area_m2
                     / (days * 86400.0), index=monthly_mm.index)


def _step_starts(step_days: int) -> np.ndarray:
    return np.arange(MELT_START_DOY, MELT_END_DOY - step_days + 2, step_days)


def runoff_as_discharge(monthly_mm: pd.Series, discharge: pd.Series,
                        step_days: int = 7, lag_steps: int = 1,
                        area_m2: float = CATCHMENT_AREA_M2) -> pd.DataFrame:
    """Apply the naive baseline: every millimetre of runoff reaches the gauge.

    No fitting, no gauge record used — each step simply takes the flow of the
    month it falls in. This is what the use case is worth measuring against,
    because it is what you would do with the climate model alone.

    Returns the same observed/predicted frame as `leave_one_year_out`, so
    `profile_skill` reads it unchanged.
    """
    flow = monthly_flow(monthly_mm, area_m2)
    observed = step_matrix(discharge, step_days)
    starts = _step_starts(step_days)
    middle = starts + step_days // 2
    # Shift by one step for the time water takes to reach the gauge, so the
    # comparison is against a fair version of the naive assumption rather than
    # one that is wrong about timing as well as magnitude.
    months = [(pd.Timestamp(2015, 1, 1)
               + pd.Timedelta(days=int(d) - 1 - step_days * lag_steps)).month
              for d in middle]

    years = [y for y in observed.index if y in set(flow.index.year)]
    predicted = pd.DataFrame(index=years, columns=observed.columns, dtype=float)
    for year in years:
        season = flow[flow.index.year == year]
        by_month = {ts.month: value for ts, value in season.items()}
        predicted.loc[year] = [by_month.get(m, 0.0) for m in months]
    return pd.concat({"observed": observed.loc[years], "predicted": predicted},
                     axis=1)


def season_shape_profiles(monthly_mm: pd.Series, discharge: pd.Series,
                          step_days: int = 7, lags: tuple[int, ...] = (0, 1, 2, 3),
                          area_m2: float = CATCHMENT_AREA_M2) -> pd.DataFrame:
    """Use the shape of the climate model's season, not just its total.

    `ProfileModel` gives every season the same shape, scaled by one number.
    This instead interpolates the monthly runoff across the steps, shifts it by
    a lag to allow for the time water takes to reach the gauge, and fits a
    single straight line from that to measured discharge.

    The lag and the line are both chosen inside each leave-one-year-out fold,
    so the held-out year never informs its own prediction.

    Returns the same observed/predicted frame as `leave_one_year_out`.
    """
    flow = monthly_flow(monthly_mm, area_m2)
    observed = step_matrix(discharge, step_days)
    starts = _step_starts(step_days)
    middle = starts + step_days // 2
    years = [y for y in observed.index if y in set(flow.index.year)]

    def shape(year: int, lag: int) -> np.ndarray:
        season = flow[flow.index.year == year]
        if not len(season):
            return np.full(len(starts), np.nan)
        # zero at the shoulders so the interpolation ramps in and out
        knots_x = [_doy(year, MELT_MONTHS[0] - 1)]
        knots_y = [0.0]
        for ts, value in season.items():
            knots_x.append(_doy(year, ts.month))
            knots_y.append(float(value))
        knots_x.append(_doy(year, MELT_MONTHS[-1] + 1))
        knots_y.append(0.0)
        return np.interp(middle - step_days * lag, knots_x, knots_y)

    shapes = {lag: np.array([shape(y, lag) for y in years]) for lag in lags}
    truth = observed.loc[years].to_numpy(dtype=float)
    predicted = pd.DataFrame(index=years, columns=observed.columns, dtype=float)

    for i, year in enumerate(years):
        others = np.arange(len(years)) != i
        best = None
        for lag, values in shapes.items():
            x = values[others].ravel()
            y = truth[others].ravel()
            keep = np.isfinite(x) & np.isfinite(y)
            if keep.sum() < 3:
                continue
            line = np.polyfit(x[keep], y[keep], 1)
            error = np.sqrt(np.mean((np.polyval(line, x[keep]) - y[keep]) ** 2))
            if best is None or error < best[0]:
                best = (error, lag, line)
        if best is None:
            continue
        _, lag, line = best
        predicted.loc[year] = np.clip(np.polyval(line, shapes[lag][i]), 0.0, None)

    return pd.concat({"observed": observed.loc[years], "predicted": predicted},
                     axis=1)


def _doy(year: int, month: int) -> float:
    """Day of year at the middle of a month, for interpolation knots."""
    month = min(max(month, 1), 12)
    stamp = pd.Timestamp(year=year, month=month, day=15)
    return float((stamp - pd.Timestamp(year=year, month=1, day=1)).days + 1)


def collapsed(frame: pd.DataFrame, summary: str, decimals: int = 2) -> None:
    """Show a table behind a click, so detail is available but not in the way."""
    from IPython.display import HTML, display

    display(HTML(f"<details><summary style='cursor:pointer'>{summary}</summary>"
                 f"{frame.round(decimals).to_html()}</details>"))


# --------------------------------------------------------------------------
# The catchment outline
# --------------------------------------------------------------------------

def load_basin(source: str | Path) -> list[np.ndarray]:
    """Read the catchment outline, as rings of (x, y), the exterior first.

    Written out from the project's `SurfaceBasin.shp` (EPSG:3413, the Watson
    polygon) so that drawing the boundary, or asking which mesh nodes fall
    inside it, needs no GIS library here.
    """
    frame = pd.read_csv(source)
    return [frame.loc[frame["ring"] == n, ["x", "y"]].to_numpy(dtype=float)
            for n in sorted(frame["ring"].unique())]


def basin_nodes(vertices: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
    """Which mesh nodes fall inside the catchment.

    The SPDE mesh reaches well beyond the catchment on purpose: a fit run to
    the edge of its own domain is distorted there, so the mesh is given a
    margin. Those outer nodes belong to the fit. They do not belong to a mean
    described as a catchment mean, which is what this mask is for.
    """
    from matplotlib.path import Path as MplPath

    xy = np.asarray(vertices, dtype=float)[:, :2]
    inside = MplPath(rings[0]).contains_points(xy)
    for hole in rings[1:]:
        inside &= ~MplPath(hole).contains_points(xy)
    return inside


def basin_path(rings: list[np.ndarray], scale: float = 1.0) -> "object":
    """Join the outline rings into one matplotlib path, holes included."""
    from matplotlib.path import Path as MplPath

    return MplPath.make_compound_path(
        *[MplPath(np.asarray(ring, dtype=float) * scale, closed=True)
          for ring in rings])


def mesh_map(ax: "object", vertices: np.ndarray, values: np.ndarray,
             rings: list[np.ndarray] | None = None,
             triangles: np.ndarray | None = None, cmap: object = "viridis",
             edges: bool = False, edge_colour: str = "white",
             edge_alpha: float = 0.55, outline: str = "#55606e",
             margin_km: float = 4.0) -> "object":
    """Draw a fitted field over the mesh, clipped to the catchment.

    The field is continuous and the mesh nodes are only where it is evaluated,
    so it is drawn as a surface rather than as points. Clipping to the outline
    keeps the picture to the catchment the use case is about; without it the
    picture covers roughly twice the area, most of it the mesh's margin.

    Pass `triangles` - the module's own triangulation, as node index triples -
    to draw the mesh the fit actually used. Without it the nodes are
    re-triangulated here, which matches on this mesh but is a reconstruction.
    `edges=True` draws the triangulation over the field, which is worth it
    where the point of the figure is that everything lands on one mesh. The
    edges have to stay legible without competing with the field, so the colour
    follows the ramp underneath: white over a mostly dark one, grey over a
    mostly pale one.

    Coordinates are divided by a thousand, so the axes are in kilometres.
    """
    import matplotlib.tri as mtri
    from matplotlib.patches import PathPatch

    xy = np.asarray(vertices, dtype=float)[:, :2] / 1000.0
    triangulation = mtri.Triangulation(
        xy[:, 0], xy[:, 1],
        None if triangles is None else np.asarray(triangles, dtype=int))
    surface = ax.tripcolor(triangulation, np.asarray(values, dtype=float),
                           cmap=cmap, shading="gouraud")
    ax.set_aspect("equal")
    drawn = [surface]
    if edges:
        drawn.extend(ax.triplot(triangulation, color=edge_colour, lw=0.35,
                                alpha=edge_alpha, zorder=3))
    if rings:
        path = basin_path(rings, scale=1 / 1000.0)
        for artist in drawn:
            artist.set_clip_path(PathPatch(path, transform=ax.transData,
                                           facecolor="none", edgecolor="none"))
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=outline,
                               lw=1.0, zorder=5))
        x, y = path.vertices[:, 0], path.vertices[:, 1]
        ax.set_xlim(x.min() - margin_km, x.max() + margin_km)
        ax.set_ylim(y.min() - margin_km, y.max() + margin_km)
    return surface


def month_axis(ax: "object", months: tuple = MELT_MONTHS) -> None:
    """Label an axis measured in days of the year with month names instead.

    A melt season is read in months rather than in the 121st to the 266th day
    of the year, and the day a step happens to start on means nothing to a
    reader. The ticks sit on the first of each month, so the space between two
    of them is the month itself.
    """
    stamps = [pd.Timestamp(2015, month, 1) for month in (*months, months[-1] + 1)]
    ax.set_xticks([stamp.dayofyear for stamp in stamps])
    ax.set_xticklabels([stamp.strftime("%b") for stamp in stamps])
    ax.set_xlabel("")


def headroom(ax: "object", fraction: float = 0.22) -> None:
    """Leave space above the data, so a legend at the top does not sit on it."""
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, bottom + (top - bottom) * (1.0 + fraction))


def label_points(ax: "object", x: object, y: object, labels: object,
                 fontsize: int = 8, colour: str = "#666666") -> None:
    """Name every point, nudged so that the names do not land on each other.

    Each label is tried in a few positions around its point and keeps the
    first that is clear of the labels already placed. Where a cluster leaves
    nowhere free, the last position is kept rather than the label dropped.
    """
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    offsets = [(4, 3), (4, -11), (-4, 3), (-4, -11), (0, 10), (0, -16)]
    placed = []
    for xi, yi, text in zip(np.asarray(x), np.asarray(y), labels, strict=False):
        note = ax.annotate(str(text), (xi, yi), xytext=offsets[0],
                           textcoords="offset points", fontsize=fontsize,
                           color=colour)
        for dx, dy in offsets:
            note.set_position((dx, dy))
            note.set_ha("left" if dx >= 0 else "right")
            box = note.get_window_extent(renderer)
            if not any(box.overlaps(other) for other in placed):
                break
        placed.append(box)


# --------------------------------------------------------------------------
# The season's shape, rather than its size
#
# `ProfileModel` gives every season the same shape, scaled by one number. This
# reads the climate model's own within-season curve instead, and fits a single
# relationship from it to measured discharge. It is the more accurate of the
# two on this catchment, and it needs a monthly curve rather than a number, so
# the two have different uses rather than one replacing the other.
# --------------------------------------------------------------------------

def season_shape(flow: pd.Series, year: int, step_days: int, lag: int) -> np.ndarray:
    """One season's runoff curve, sampled at the middle of each step."""
    starts = _step_starts(step_days)
    middle = starts + step_days // 2
    season = flow[flow.index.year == year]
    if not len(season):
        return np.full(len(starts), np.nan)
    # zero at the shoulders so the interpolation ramps in and out
    knots_x = [_doy(year, MELT_MONTHS[0] - 1)]
    knots_y = [0.0]
    for ts, value in season.items():
        knots_x.append(_doy(year, ts.month))
        knots_y.append(float(value))
    knots_x.append(_doy(year, MELT_MONTHS[-1] + 1))
    knots_y.append(0.0)
    return np.interp(middle - step_days * lag, knots_x, knots_y)


def shape_matrix(monthly_mm: pd.Series, years: Years, step_days: int = 7, lag: int = 0,
                 area_m2: float = CATCHMENT_AREA_M2) -> pd.DataFrame:
    """Runoff curves for several seasons, rows years and columns day-of-year."""
    flow = monthly_flow(monthly_mm, area_m2)
    starts = _step_starts(step_days)
    return pd.DataFrame({int(y): season_shape(flow, int(y), step_days, lag)
                         for y in years}, index=starts).T


@dataclass
class ShapeModel:
    """Discharge from the shape of the climate model's season."""

    starts: np.ndarray
    step_days: int
    lag: int
    slope: float
    intercept: float
    residual_sd: float
    sd_intercept: float
    sd_slope: float
    dof: int
    n: int
    xbar: float
    sxx: float
    years: np.ndarray
    r2: float
    monthly: pd.Series
    area_m2: float
    extra_names: list
    extra_coefficients: np.ndarray
    extras: pd.DataFrame | None

    @classmethod
    def fit(cls, discharge: pd.Series, monthly_mm: pd.Series, step_days: int = 7,
            lags: tuple[int, ...] = (0, 1, 2, 3),
            extras: pd.DataFrame | None = None,
            area_m2: float = CATCHMENT_AREA_M2) -> "ShapeModel":
        """Fit the curve, and optionally extra per-season predictors.

        `extras` is one value per season per predictor - a lake-intensity
        coefficient, say - held constant across the steps of that season. The
        curve carries the within-season shape, the extras shift the level.
        """
        observed = step_matrix(discharge, step_days)
        flow = monthly_flow(monthly_mm, area_m2)
        years = np.array([y for y in observed.index if y in set(flow.index.year)])
        extra_frame = None
        if extras is not None:
            extra_frame = (extras.to_frame() if isinstance(extras, pd.Series)
                           else pd.DataFrame(extras))
            years = np.array([y for y in years if y in extra_frame.index])
            extra_frame = extra_frame.loc[years]
        truth = observed.loc[years].to_numpy(dtype=float)
        n_steps = truth.shape[1]

        best = None
        for lag in lags:
            shapes = shape_matrix(monthly_mm, years, step_days, lag, area_m2)
            columns = [np.ones(truth.size), shapes.to_numpy(dtype=float).ravel()]
            if extra_frame is not None:
                for name in extra_frame.columns:
                    columns.append(np.repeat(
                        extra_frame[name].to_numpy(dtype=float), n_steps))
            design = np.column_stack(columns)
            y = truth.ravel()
            keep = np.isfinite(y) & np.isfinite(design).all(axis=1)
            if keep.sum() < design.shape[1] + 1:
                continue
            beta, *_ = np.linalg.lstsq(design[keep], y[keep], rcond=None)
            residuals = y[keep] - design[keep] @ beta
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            if best is None or rmse < best[0]:
                best = (rmse, lag, beta[1], beta[0], design[keep, 1], y[keep],
                        residuals, beta[2:])

        rmse, lag, slope, intercept, x, y, residuals, extra_beta = best
        dof = max(len(x) - 2, 1)
        denominator = float(((y - y.mean()) ** 2).sum())
        # The scatter is not constant - a week predicted at 1500 m3/s misses by
        # far more than a week predicted at 50. A single pooled sd would be too
        # wide at the shoulders of the season and too narrow at the peak, so
        # fit the spread as a function of the predicted flow. The sqrt(pi/2)
        # converts mean absolute deviation to a standard deviation.
        fitted = y - residuals
        sd_slope, sd_intercept = np.polyfit(fitted, np.abs(residuals), 1)
        scale = float(np.sqrt(np.pi / 2.0))
        return cls(starts=np.asarray(observed.columns, dtype=int),
                   step_days=step_days, lag=int(lag), slope=float(slope),
                   intercept=float(intercept),
                   residual_sd=float(np.sqrt((residuals ** 2).sum() / dof)),
                   sd_intercept=float(sd_intercept * scale),
                   sd_slope=float(sd_slope * scale), dof=int(dof), n=int(len(x)), xbar=float(x.mean()),
                   sxx=float(((x - x.mean()) ** 2).sum()), years=years,
                   r2=float(1 - (residuals ** 2).sum() / denominator) if denominator else np.nan,
                   monthly=monthly_mm, area_m2=area_m2,
                   extra_names=list(extra_frame.columns) if extra_frame is not None else [],
                   extra_coefficients=np.asarray(extra_beta, dtype=float),
                   extras=extra_frame)

    # ------------------------------------------------------------ predict --
    def shape(self, year: int) -> np.ndarray:
        """Give the runoff curve this model would use for a season."""
        flow = monthly_flow(self.monthly, self.area_m2)
        return season_shape(flow, int(year), self.step_days, self.lag)

    def _as_shape(self, values: Values) -> np.ndarray:
        if np.isscalar(values):
            return self.shape(values)
        return np.asarray(values, dtype=float).ravel()

    def _offset(self, year: int | None) -> float:
        """Contribution of the extra per-season predictors, if any."""
        if not self.extra_names or self.extras is None or year is None:
            return 0.0
        if year not in self.extras.index:
            return float(self.extra_coefficients @ self.extras.mean().to_numpy())
        return float(self.extra_coefficients @ self.extras.loc[year].to_numpy(float))

    def profile(self, values: Values,
                extras: pd.DataFrame | np.ndarray | None = None) -> pd.Series:
        """Discharge through one melt season, m3/s. Takes a year or a curve.

        An `extras` row given here is used as it stands, which is what a
        held-out season needs: its own extra predictors are known, but the
        model was fitted without them, so looking the year up would silently
        fall back to the training mean and the prediction would not depend on
        them at all.
        """
        curve = self._as_shape(values)
        if extras is not None:
            offset = float(self.extra_coefficients
                           @ np.asarray(extras, dtype=float).ravel())
        elif np.isscalar(values):
            offset = self._offset(int(values))
        else:
            offset = 0.0
        return pd.Series(
            np.clip(self.intercept + offset + self.slope * curve, 0.0, None),
            index=self.starts, name="m3s")

    def profiles(self, years: Years) -> pd.DataFrame:
        """One profile per season: rows are years, columns day-of-year."""
        index = getattr(years, "index", years)
        return pd.DataFrame({int(y): self.profile(int(y)) for y in index}).T

    def prediction_se(self, values: Values) -> np.ndarray:
        """Give the standard error of a new observation, per step.

        Simple-regression prediction error: the residual scatter, widened by
        the uncertainty of the fitted line and by how far the runoff value sits
        from the mean of the values it was fitted on.
        """
        curve = self._as_shape(values)
        offset = self._offset(int(values)) if np.isscalar(values) else 0.0
        fitted = np.clip(self.intercept + offset + self.slope * curve, 0.0, None)
        local_sd = np.clip(self.sd_intercept + self.sd_slope * fitted,
                           0.05 * self.residual_sd, None)
        leverage = 1.0 / self.n + (curve - self.xbar) ** 2 / self.sxx
        return local_sd * np.sqrt(1.0 + np.clip(leverage, 0.0, None))

    def leverage(self, values: Values) -> float:
        """Say how wide this season's band is against a typical one."""
        here = float(np.nanmean(self.prediction_se(values)))
        centre = float(np.nanmean(
            [np.nanmean(self.prediction_se(int(y))) for y in self.years]))
        return here / centre if centre else 1.0

    def _multiplier(self, sigmas: float | None, level: float) -> float:
        if sigmas is not None:
            level = math.erf(sigmas / math.sqrt(2.0))
        return _t_quantile(0.5 + level / 2.0, self.dof)

    def band(self, values: Values, sigmas: float | None = None,
             level: float = 0.68) -> pd.DataFrame:
        """Prediction interval around one season, at one coverage."""
        centre = self.profile(values)
        spread = pd.Series(self.prediction_se(values) * self._multiplier(sigmas, level),
                           index=self.starts)
        return pd.DataFrame({"predicted": centre,
                             "lower": (centre - spread).clip(lower=0),
                             "upper": centre + spread})

    def bands(self, values: Values,
              levels: Sequence[float] = (0.68, 0.95)) -> pd.DataFrame:
        """Several coverages at once: columns lower_68, upper_68, lower_95, ..."""
        out = pd.DataFrame({"predicted": self.profile(values)})
        for level in levels:
            band = self.band(values, level=level)
            tag = f"{round(level * 100)}"
            out[f"lower_{tag}"] = band["lower"]
            out[f"upper_{tag}"] = band["upper"]
        return out

    def __str__(self) -> str:
        """Summarise the fit in one line."""
        extra = (f" plus {', '.join(self.extra_names)}" if self.extra_names else "")
        return (f"{len(self.starts)} steps of {self.step_days} days, fitted across "
                f"{len(self.years)} seasons ({self.years.min()}-{self.years.max()}) "
                f"on the runoff curve with a {self.lag}-step lag{extra}. "
                f"r2 = {self.r2:.2f}, residual sd {self.residual_sd:.0f} m3/s.")


def shape_leave_one_year_out(discharge: pd.Series, monthly_mm: pd.Series,
                             step_days: int = 7, extras: pd.DataFrame | None = None,
                             lags: tuple = (0, 1, 2, 3),
                             area_m2: float = CATCHMENT_AREA_M2) -> pd.DataFrame:
    """Predict each season from a shape model that never saw that season.

    The companion to `leave_one_year_out` for `ShapeModel`. The lag and the
    line are both chosen inside each fold, and any extra per-season predictor
    is refitted there too, so nothing about the held-out year reaches the model
    that predicts it.

    Returns the same observed/predicted frame the other models return.
    """
    observed = step_matrix(discharge, step_days)
    flow_years = set(monthly_mm.index.year)
    years = [year for year in observed.index if year in flow_years]
    if extras is not None:
        years = [year for year in years if year in set(extras.index)]
    predicted = pd.DataFrame(index=years, columns=observed.columns, dtype=float)

    for year in years:
        others = [other for other in years if other != year]
        model = ShapeModel.fit(discharge[discharge.index.year != year], monthly_mm,
                               step_days, lags=lags,
                               extras=None if extras is None else extras.loc[others],
                               area_m2=area_m2)
        row = None if extras is None else extras.loc[[year]]
        predicted.loc[year] = model.profile(year, extras=row).to_numpy()

    return pd.concat({"observed": observed.loc[years], "predicted": predicted},
                     axis=1)

