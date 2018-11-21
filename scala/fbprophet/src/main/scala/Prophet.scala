import Domain.{Holiday, Regressor, Seasonality}
import Models.ProphetStanModel
import com.cibo.scalastan.RunMethod.OptimizeAlgorithm
import com.cibo.scalastan.{RunMethod, StanResults}

import scala.collection.mutable

/**
  *
  * @param growth                : 'linear' or 'logistic' to specify a linear or logistic trend.
  * @param changePoints          : List of epochs at which to include potential changepoints.
  *                              If not specified, potential changepoints are selected automatically.
  * @param nChangePoints         : Number of potential changepoints to include. Not used
  *                              if input `changepoints` is supplied. If `changepoints` is not supplied,
  *                              then nChangePoints potential changepoints are selected uniformly from
  *                              the first `changePointRange` proportion of the history.
  * @param changePointRange      : Proportion of history in which trend changepoints will
  *                              be estimated. Defaults to 0.8 for the first 80%. Not used if
  *                              `changepoints` is specified.
  *                              Not used if input `changepoints` is supplied.
  * @param yearlySeasonality     :Fit yearly seasonality.
  *                              Can be 'auto', True, False, or a number of Fourier terms to generate.
  * @param weeklySeasonality     Fit weekly seasonality.
  *                              Can be 'auto', True, False, or a number of Fourier terms to generate.
  * @param dailySeasonality      : Fit daily seasonality.
  *                              Can be 'auto', True, False, or a number of Fourier terms to generate.
  * @param holidays              : Case class with columns holiday (string) and ds (date type)
  *                              and optionally columns lower_window and upper_window which specify a
  *                              range of days around the date to be included as holidays.
  *                              lower_window=-2 will include 2 days prior to the date as holidays. Also
  *                              optionally can have a column prior_scale specifying the prior scale for
  *                              that holiday.
  * @param appendHolidays        : country name or abbreviation; must be string
  * @param seasonalityMode       : 'additive' (default) or 'multiplicative'.
  * @param seasonalityPriorScale : Parameter modulating the strength of the
  *                              seasonality model. Larger values allow the model to fit larger seasonal
  *                              fluctuations, smaller values dampen the seasonality. Can be specified
  *                              for individual seasonalities using add_seasonality.
  * @param holidayPriorScale     : Parameter modulating the strength of the holiday
  *                              components model, unless overridden in the holidays input.
  * @param changePointPriorScale : Parameter modulating the flexibility of the
  *                              automatic changepoint selection. Large values will allow many
  *                              changepoints, small values will allow few changepoints
  * @param mcmcSamples           : Integer, if greater than 0, will do full Bayesian inference
  *                              with the specified number of MCMC samples. If 0, will do MAP
  *                              estimation.
  * @param intervalWidth         : Float, width of the uncertainty intervals provided
  *                              for the forecast. If mcmc_samples=0, this will be only the uncertainty
  *                              in the trend using the MAP estimate of the extrapolated generative
  *                              model. If mcmc.samples>0, this will be integrated over all model
  * @param uncertaintySamples    : Number of simulated draws used to estimate
  *                              uncertainty intervals.
  */
class Prophet(growth: String = "linear",
              changePoints: Seq[Double] = Seq.empty,
              var nChangePoints: Int = 25,
              changePointRange: Double = 0.8,
              yearlySeasonality: Any = "auto",
              weeklySeasonality: Any = "auto",
              dailySeasonality: Any = "auto",
              holidays: Option[Holiday] = None,
              appendHolidays: Option[String] = None,
              seasonalityMode: String = "additive",
              seasonalityPriorScale: Double = 10.0,
              holidayPriorScale: Double = 10.0,
              changePointPriorScale: Double = 0.05,
              mcmcSamples: Int = 0,
              intervalWidth: Double = 0.80, // Number of regressors
              uncertaintySamples: Int = 1000 // Capacities for logistic trend
             ) extends ProphetStanModel {

  if (changePoints.nonEmpty) nChangePoints = changePoints.length
  validate_inputs()

  val changepoints_t: mutable.Seq[Double] = mutable.Seq.empty
  val params: Seq[(String, Stream[Any])] => mutable.Map[String, Stream[Any]] = collection.mutable.Map[String, Stream[Any]]
  var seasonalities: mutable.Seq[Seasonality] = mutable.Seq.empty
  var regressors: mutable.Seq[Regressor] = mutable.Seq.empty

  // Set during fitting
  var history: collection.mutable.Map[String, Stream[Any]] = mutable.Map.empty
  var start: Double = 0
  var y_scale: Double = 0
  var logistic_floor: Boolean = false
  var t_scale: Double = 0

  /*
      Validates the inputs to Prophet
   */
  def validate_inputs() {
    if (!(growth == "linear" || growth == "logistic")) {
      throw new RuntimeException("Parameter 'growth' should be 'linear' or 'logistic'.")
    }
    if ((changePointRange < 0) || (changePointRange > 1)) {
      throw new RuntimeException("Parameter 'changePointRange' must be in [0, 1]")
    }
    if (!(seasonalityMode == "additive" || seasonalityMode == "multiplicative")) {
      throw new RuntimeException("seasonalityMode must be 'additive' or 'multiplicative'")
    }
  }

  /**
    * Add a seasonal component with specified period, number of Fourier
    * components, and prior scale.
    *
    * Increasing the number of Fourier components allows the seasonality to
    * change more quickly (at risk of overfitting). Default values for yearly
    * and weekly seasonalities are 10 and 3 respectively.
    *
    * Increasing prior scale will allow this seasonality component more
    * flexibility, decreasing will dampen it. If not provided, will use the
    * seasonality_prior_scale provided on Prophet initialization (defaults
    * to 10).
    *
    * Mode can be specified as either 'additive' or 'multiplicative'. If not
    * specified, self.seasonality_mode will be used (defaults to additive).
    * Additive means the seasonality will be added to the trend,
    * multiplicative means it will multiply the trend.
    *
    * @param name          :string name of the seasonality component.
    * @param period        : float number of days in one period.
    * @param fourier_order : int number of Fourier components to use.
    * @param prior_scale   : optional float prior scale for this component.
    * @param mode          : optional 'additive' or 'multiplicative'
    */


  def add_seasonality(name: String, period: Double, fourier_order: Int, prior_scale: Option[Double] = None, mode: Option[String] = None) {

    if (history.nonEmpty) throw new RuntimeException("Seasonality must be added prior to model fitting.")
    if (!Seq("daily", "weekly", "yearly").contains(name)) {
      // Allow overwriting built-in seasonalities
      validate_column_name(name, check_seasonalities = false)
    }
    val ps = prior_scale.getOrElse(seasonalityPriorScale)
    if (ps <= 0) throw new RuntimeException("Prior scale must be >0")

    val md = mode.getOrElse(seasonalityMode)
    if (!Seq("additive", "multiplicative").contains(md)) throw new RuntimeException("mode must be 'additive' or 'multiplicative'")
    seasonalities = seasonalities :+ Seasonality(name, period, fourier_order, Some(ps), mode)
  }

  /**
    *
    * @param name                :String
    * @param check_holidays      :Boolean, check if name already used for holiday
    * @param check_seasonalities :Boolean, check if name already used for seasonality
    * @param check_regressors    :Boolean, check if name already used for regressor
    * @return
    */
  def validate_column_name(name: String, check_holidays: Boolean = true,
                           check_seasonalities: Boolean = true, check_regressors: Boolean = true) = {

    if (name.contains('_delim_)) throw new RuntimeException("Name cannot contain '_delim_'")
    var reserved_names = Seq("trend", "additive_terms", "daily", "weekly", "yearly", "holidays", "zeros", "extra_regressors_additive", "yhat", "extra_regressors_multiplicative", "multiplicative_terms", "ds", "y", "cap", "floor", "y_scaled", "cap_scaled")
    val rn_l = reserved_names.map(a => a + "_lower")
    val rn_u = reserved_names.map(a => a + "_upper")
    reserved_names = reserved_names ++ rn_l ++ rn_u
    if (reserved_names.contains(name)) {
      throw new RuntimeException("Name " + name + " is reserved")
    }
    if (check_seasonalities && seasonalities.map(s => s.name).contains(name)) throw new RuntimeException("Name " + name + " is already used for a seasonality")

    if (check_regressors && regressors.map(s => s.name).contains(name)) throw new RuntimeException("Name " + name + " is already used for a regressor")
  }

  /**
    * Add an additional regressor to be used for fitting and predicting.
    *
    * The dataframe passed to `fit` and `predict` will have a column with the
    * specified name to be used as a regressor. When standardize='auto', the
    * regressor will be standardized unless it is binary. The regression
    * coefficient is given a prior with the specified scale parameter.
    * Decreasing the prior scale will add additional regularization. If no
    * prior scale is provided, self.holidays_prior_scale will be used.
    * Mode can be specified as either 'additive' or 'multiplicative'. If not
    * specified, self.seasonality_mode will be used. 'additive' means the
    * effect of the regressor will be added to the trend, 'multiplicative'
    * means it will multiply the trend.
    *
    * @param name        : string name of the regressor.
    * @param standardize : optional, specify whether this regressor will be
    *                    standardized prior to fitting. Can be 'auto' (standardize if not
    *                    binary), True, or False.
    * @param priorScale  : optional float scale for the normal prior. If not
    *                    provided, self.holidays_prior_scale will be used.
    * @param mode        : optional, 'additive' or 'multiplicative'. Defaults to
    *             self.seasonality_mode.
    */
  def add_regressor(name: String, standardize: Option[Boolean], priorScale: Option[Double] = None, mode: Option[String] = None) {
    if (history.nonEmpty) throw new RuntimeException("Regressors must be added prior to model fitting.")
    validate_column_name(name, check_regressors = false)
    val ps = priorScale.getOrElse(seasonalityPriorScale)
    if (ps <= 0) throw new RuntimeException("Prior scale must be >0")

    val md = mode.getOrElse(seasonalityMode)
    if (!Seq("additive", "multiplicative").contains(md)) throw new RuntimeException("mode must be 'additive' or 'multiplicative'")
    regressors = regressors :+ Domain.Regressor(name, 0, 1, standardize, Some(ps), Some(md))
  }

  def fit(events: Map[String, Stream[Any]]): StanResults = {

    if (!events.contains("y") || !events.contains("ds"))
      throw new RuntimeException("History must have keys 'ds' and 'y' with the dates and values respectively")

    if (history.nonEmpty) {
      throw new RuntimeException("Prophet object can only be fit once.'Instantiate a new object.")
    }

    history ++= events
    if (history("y").length < 2) throw new RuntimeException("events has less than 2 non-NaN rows.")


    setup_history(history, initialize_scales = true)
    fourier_series(getDoubleSequence(history("t")), period = 7, series_order = 4)
    val seq: Seq[Seq[Double]] = Seq.fill(history("ds").length)(Seq(0))
    val model = super.compile
      .withInitialValue(k, 0.07253355189017519)
      .withInitialValue(m, 0.1738551377932589)
      .withInitialValue(delta, Seq(0d))
      .withInitialValue(beta, Seq(0d))
      .withInitialValue(sigma_obs, 1d)
      .withData(cap, Seq.fill(history("ds").length)(0d))
      .withData(T, history("ds").length)
      .withData(K, 1)
      .withData(S, 1)
      .withData(y, getDoubleSequence(history("y_scaled")))
      .withData(t, getDoubleSequence(history("t")))
      .withData(t_change, Seq.fill(1)(0d))
      .withData(X, seq)
      .withData(sigmas, Seq(1.0))
      .withData(tau, changePointPriorScale)
      .withData(trend_indicator, 0)
      .withData(s_a, Seq.fill(1)(0d))
      .withData(s_m, Seq.fill(1)(0d))

    try {
      model.run(cache = false, method = RunMethod.Optimize())
    } catch {
      case _: Exception => model.run(cache = false, method = RunMethod.Optimize(algorithm = Newton()))
    }
  }

  /**
    * //    * Provides Fourier series components with the specified frequency
    * and order.
    *
    * @param ds           : Sequence containing timestamps
    * @param period       :  Number of days of the period.
    * @param series_order : Number of components.
    */
  def fourier_series(ds: Stream[Double], period: Float, series_order: Int) {
    ds.map(a => (1 to series_order).map(
      i => (math.cos(2.0 * (i + 1) * math.Pi * a / period), math.sin(2.0 * (i + 1) * math.Pi * a / period))))
      .map(a => a.flatten { case (p, q) => Seq(p, q) })
  }

  private def setup_history(events: mutable.Map[String, Stream[Any]], initialize_scales: Boolean = false): Unit = {

    // TODO: Add different timestamps support, assuming long for now
    // TODO: Add timestamps sort support, assuming sorted for now
    initializeScales(initialize_scales)
    history.put("t", getDoubleSequence(history("ds")).map(a => (a - start) / t_scale))
    history.put("y_scaled", getDoubleSequence(history("y")).map(a => a / y_scale))
    history ++= setAutoSeasonalities()


  }

  private def initializeScales(initialize_scales: Boolean): Unit = {
    if (!initialize_scales) return

    y_scale = getDoubleSequence(history("y")).map(a => Math.abs(a)).max
    if (y_scale == 0) y_scale = 1

    start = getDoubleSequence(history("ds")).min
    t_scale = getDoubleSequence(history("ds")).max - start
  }

  private def getDoubleSequence(any: Stream[Any]): Stream[Double] = {
    any.map(a => a.asInstanceOf[Double])
  }

  private def setAutoSeasonalities(): Map[String, Stream[Double]] = {
    Map("zeros" -> Stream.fill(history.keySet.size)(0))
  }

  // Needs dummy
  case class Newton() extends OptimizeAlgorithm("newton") {
    def arguments: Seq[String] = build(("DUMMY", "DUMMY", "DUMMY"))
  }

}