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
  val params: Seq[(String, Seq[Any])] => mutable.Map[String, Seq[Any]] = collection.mutable.Map[String, Seq[Any]]
  var history: collection.mutable.Map[String, Seq[Any]] = mutable.Map.empty
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

  def fit(events: Map[String, Seq[Any]]): StanResults = {

    if (!events.contains("y") || !events.contains("ds"))
      throw new RuntimeException("History must have keys 'ds' and 'y' with the dates and values respectively")

    if (history.nonEmpty) {
      throw new RuntimeException("Prophet object can only be fit once.'Instantiate a new object.")
    }

    history ++= events
    if (history("y").length < 2) throw new RuntimeException("events has less than 2 non-NaN rows.")


    setup_history(history, initialize_scales = true)

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

  private def setup_history(events: mutable.Map[String, Seq[Any]], initialize_scales: Boolean = false): Unit = {

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

  private def getDoubleSequence(any: Seq[Any]): Seq[Double] = {
    any.map(a => a.asInstanceOf[Double])
  }

  private def setAutoSeasonalities(): Map[String, Seq[Double]] = {
    Map("zeros" -> Seq.fill(history.keySet.size)(0))
  }

  // Needs dummy
  case class Newton() extends OptimizeAlgorithm("newton") {
    def arguments: Seq[String] = build(("DUMMY", "DUMMY", "DUMMY"))
  }

}