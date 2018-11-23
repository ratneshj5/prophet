import java.util.concurrent.TimeUnit

import Domain._
import Models.ProphetStanModel
import Prophet._
import com.cibo.scalastan.RunMethod.OptimizeAlgorithm
import com.cibo.scalastan.{RunMethod, StanResults}

import scala.collection.mutable

/**
  * Prophet forecaster.
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
  * @param seasonalityMode       : 'additive' (default) or 'multiplicative'.
  * @param seasonalityPriorScale : Parameter modulating the strength of the
  *                              seasonality model. Larger values allow the model to fit larger seasonal
  *                              fluctuations, smaller values dampen the seasonality. Can be specified
  *                              for individual seasonalities using add_seasonality.
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
              changePoints: Stream[Double] = Stream.empty,
              var nChangePoints: Int = 25,
              changePointRange: Double = 0.8,
              yearlySeasonality: Any = "auto",
              weeklySeasonality: Any = "auto",
              dailySeasonality: Any = "auto",
              seasonalityMode: String = "additive",
              seasonalityPriorScale: Double = 10.0,
              changePointPriorScale: Double = 0.05,
              mcmcSamples: Int = 0,
              intervalWidth: Double = 0.80, // Number of regressors
              uncertaintySamples: Int = 1000 // Capacities for logistic trend
             ) extends ProphetStanModel {
  var changepoints_t: Stream[Double] = Stream.empty
  if (changePoints.nonEmpty) {
    changepoints_t = changePoints
    nChangePoints = changePoints.length
  }
  validate_inputs()
  var seasonalities: mutable.Seq[Seasonality] = mutable.Seq.empty
  var regressors: mutable.Seq[Regressor] = mutable.Seq.empty

  // Set during fitting
  var history: collection.mutable.Map[String, Stream[Any]] = mutable.Map.empty
  var start: Double = 0
  var y_scale: Double = 0
  var logistic_floor: Boolean = false
  var t_scale: Double = 0

  //Set during fitting
  var parameter: Parameter = Parameter(k = 0, m = 0, delta = Seq.empty, sigma_obs = 0, beta = Seq.empty)


  /**
    * Fit the Prophet model.
    *
    * This sets params to contain the fitted model parameters. It is a
    * dictionary parameter names as keys and the following items:
    * k (Mx1 array): M posterior samples of the initial slope.
    * m (Mx1 array): The initial intercept.
    * delta (MxN array): The slope change at each of N changepoints.
    * beta (MxK matrix): Coefficients for K seasonality features.
    * sigma_obs (Mx1 array): Noise level.
    * Note that M=1 if MAP estimation.
    *
    * @param events : Map containing the history. Must have keys ds (epoch
    *               type) and y, the time series. If growth is 'logistic', then
    *               df must also have a key cap that specifies the capacity at
    *               each ds.
    * @return
    * The fitted Prophet object.
    */

  def fit(events: Map[String, Stream[Any]]): Prophet = {

    if (!events.contains("y") || !events.contains("ds"))
      throw new RuntimeException("History must have keys 'ds' and 'y' with the dates and values respectively")

    if (history.nonEmpty) {
      throw new RuntimeException("Prophet object can only be fit once.'Instantiate a new object.")
    }

    history ++= events
    if (history("y").length < 2) throw new RuntimeException("events has less than 2 non-NaN rows.")
    history = setup_history(history, initialize_scales = true)
    setAutoSeasonalities()
    val seasonalData = makeAllSeasonalityFeatures(history, seasonalities, regressors)
    setChangePoints()
    parameter = init_param(history, seasonalData)
    val model = super.compile
      .withInitialValue(k, parameter.k)
      .withInitialValue(m, parameter.m)
      .withInitialValue(delta, parameter.delta)
      .withInitialValue(beta, parameter.beta)
      .withInitialValue(sigma_obs, parameter.sigma_obs)
      .withData(cap, getDoubleSequence(history("cap_scaled")))
      .withData(T, history("ds").length)
      .withData(K, seasonalData.prior_scales.length)
      .withData(S, changepoints_t.length)
      .withData(y, getDoubleSequence(history("y_scaled")))
      .withData(t, getDoubleSequence(history("t")))
      .withData(t_change, changepoints_t)
      .withData(X, seasonalData.seasonalFeatures)
      .withData(sigmas, seasonalData.prior_scales)
      .withData(tau, changePointPriorScale)
      .withData(trend_indicator, if (growth == "linear") 0 else 1)
      .withData(s_a, seasonalData.s_a)
      .withData(s_m, seasonalData.s_m)

    if (growth == "linear" && getDoubleSequence(history("y")).max == getDoubleSequence(history("y_scaled")).min) {
      parameter = Parameter(parameter.k, parameter.m, parameter.delta, Math.pow(10, -9), parameter.beta)
    } else if (mcmcSamples > 0) {
      val results = model.run(cache = false, method = RunMethod.Sample(samples = mcmcSamples))
      parameter = extract_params(results)
    } else {
      try {
        val results = model.run(cache = false, method = RunMethod.Optimize(iter = 10000))
        parameter = extract_params(results)
      } catch {
        case _: Exception =>
          val results = model.run(cache = false, method = RunMethod.Optimize(iter = 10000, algorithm = Newton()))
          parameter = extract_params(results)
      }
    }

    def extract_params(results: StanResults): Parameter = {
      val _k = results.best(k)
      val _m = results.best(m)
      val _delta = results.best(delta)
      val _sigma_obs = results.best(sigma_obs)
      val _beta = results.best(beta)
      Parameter(_k, _m, _delta, _sigma_obs, _beta)
    }

    print(parameter.toString)
    this
  }

  /**
    * Creates auxiliary keys 't', 't_ix',
    * 'y_scaled', and 'cap_scaled'. These columns are used during both
    * fitting and predicting.
    *
    * @param events            : Map with ds, y, and cap if logistic growth. Any
    *                          specified additional regressors must also be present.
    * @param initialize_scales Boolean set scaling factors in self from events
    */
  private def setup_history(events: mutable.Map[String, Stream[Any]],
                            initialize_scales: Boolean = false): mutable.Map[String, Stream[Any]] = {

    // TODO: Add different timestamps support, assuming long for now
    // TODO: Add timestamps sort support, assuming sorted for now

    regressors.foreach(regressor => if (!events.contains(regressor.name)) {
      throw new RuntimeException(s"Regressor ${regressor.name} missing from events")
    })
    initializeScales(events, initialize_scales)

    if (logistic_floor) {
      if (!events.contains("floor"))
        throw new RuntimeException("Expected column 'floor'.")
    }
    else events("floor") = Stream.fill(events("ds").length)(0d)
    val typedY = getDoubleSequence(events("y"))

    val floor = getDoubleSequence(events("floor"))

    if (growth == "logistic") {
      if (!events.contains("cap"))
        throw new RuntimeException("Capacities must be supplied for logistic growth in column 'cap'")
      val typedCap = getDoubleSequence(events("cap"))
      events.put("cap_scaled", (typedCap zip floor).map(a => Math.abs(a._1 - a._2) / y_scale))
    } else {
      events.put("cap_scaled", Stream.fill(typedY.length)(0d))
    }

    events.put("t", getDoubleSequence(events("ds")).map(a => (a - start) / t_scale))
    if (events.contains("y"))
      events.put("y_scaled", (typedY zip floor).map(a => Math.abs(a._1 - a._2) / y_scale))

    regressors.foreach(
      regressor => {
        getDoubleSequence(events(regressor.name)).foreach(
          a => (a - regressor.mu) / regressor.std
        )
      }
    )
    events
  }

  /**
    * Sets model scaling factors using events.
    *
    * @param events            : Map with ds, y, and cap if logistic growth. Any
    *                          specified additional regressors must also be present.
    * @param initialize_scales : Boolean set scaling factors in self from events
    */

  private def initializeScales(events: mutable.Map[String, Stream[Any]],
                               initialize_scales: Boolean): Unit = {
    if (!initialize_scales) return
    var floor: Stream[Double] = Stream.empty
    val typedDS = getDoubleSequence(events("ds"))
    val typedY = getDoubleSequence(events("y"))

    if (growth == "logistic" && events.contains("floor")) {
      logistic_floor = true
      floor = getDoubleSequence(events("floor"))
    }
    else {
      floor = Stream.fill(typedDS.length)(0d)
    }
    y_scale = (typedY zip floor).map(a => Math.abs(a._1 - a._2)).max
    if (y_scale == 0d) y_scale = 1d

    start = typedDS.min
    t_scale = typedDS.max - start

    regressors.foreach(
      regressor => {
        var standarize: Option[Boolean] = regressor.standardize
        val stream = getDoubleSequence(events(regressor.name))
        val n_vals = stream.distinct.length
        if (n_vals < 2) standarize = Some(false)
        if (standarize.isEmpty) {
          if (stream.distinct.toSet == Set(1d, 0d)) standarize = Some(false)
          else standarize = Some(true)
        }
        if (standarize.get)
          regressor.mu = stream.sum / stream.length
        val deviation = stream.map(score => (score - regressor.mu) * (score - regressor.mu))
        regressor.std = Math.sqrt(deviation.sum / (stream.length - 1))
      }
    )
  }

  /**
    *
    * Set seasonalities that were left on auto.
    *
    * Turns on yearly seasonality if there is >=2 years of history.
    * Turns on weekly seasonality if there is >=2 weeks of history, and the
    * spacing between dates in the history is <7 days.
    * Turns on daily seasonality if there is >=2 days of history, and the
    * spacing between dates in the history is <1 day.
    */
  private def setAutoSeasonalities(): Unit = {
    val typedDS = getDoubleSequence(history("ds"))
    val first = typedDS.min
    val last = typedDS.max
    val stream: Stream[Double] = Stream(typedDS.head) #::: typedDS
    val min_dt = (typedDS zip stream).map({ case (a, b) => a - b }).filter(a => a != 0).min

    // Yearly seasonality
    val yDisable = (last - first) < TimeUnit.DAYS.toMillis(730)
    val yOrder = parseSeasonalityArgs("yearly", yearlySeasonality, yDisable, 10)
    if (yOrder > 0)
      seasonalities = seasonalities :+ Seasonality("yearly", 365.25, yOrder, seasonalityPriorScale, seasonalityMode)

    val wDisable = (last - first) < TimeUnit.DAYS.toMillis(14) || min_dt >= TimeUnit.DAYS.toMillis(7)
    val wOrder = parseSeasonalityArgs("weekly", weeklySeasonality, wDisable, 3)
    if (wOrder > 0)
      seasonalities = seasonalities :+ Seasonality("weekly", 7, wOrder, seasonalityPriorScale, seasonalityMode)

    val dDisable = (last - first) < TimeUnit.DAYS.toMillis(2) || min_dt >= TimeUnit.DAYS.toMillis(1)
    val dOrder = parseSeasonalityArgs("daily", dailySeasonality, dDisable, 4)
    if (dOrder > 0)
      seasonalities = seasonalities :+ Seasonality("daily", 1, dOrder, seasonalityPriorScale, seasonalityMode)
  }

  /**
    *
    * @param name         : string name of the seasonality component.
    * @param arg          : 'auto', True, False, or number of fourier components as provided.
    * @param autoDisable  : bool if seasonality should be disabled when 'auto'.
    * @param defaultOrder : int default fourier order
    * @return Number of fourier components, or 0 for disabled.
    */
  private def parseSeasonalityArgs(name: String, arg: Any, autoDisable: Boolean, defaultOrder: Int): Int = {
    var fourierOrder = 0
    if (arg == "auto") {
      fourierOrder = 0
      seasonalities.foreach(seasonality => {
        if (seasonality.name == name)
          logger.info(s"Found custom seasonality named $name disabling built-in $name seasonality")
      })
      if (autoDisable) {
        logger.info(s"Disabling $name seasonality. Run prophet with ${name}_seasonality=True to override this.")
      }
      else fourierOrder = defaultOrder
    }
    else if (arg == true) fourierOrder = defaultOrder
    else if (arg == false) fourierOrder = 0
    else fourierOrder = arg.toString.toInt

    fourierOrder
  }

  private def setChangePoints(): Unit = {
    val typedDS = getDoubleSequence(history("ds"))
    if (changepoints_t.nonEmpty) {
      val tooLow = changepoints_t.min < typedDS.min
      val tooHigh = changepoints_t.max < typedDS.max
      if (tooHigh || tooLow)
        throw new RuntimeException("Changepoints must fall within training data")
    } else {
      val historySize = Math.floor(typedDS.length * changePointRange).toInt
      if (nChangePoints + 1 > historySize) {
        nChangePoints = historySize - 1
        logger.info(s"nChangePoints greater than number of observations. Using $nChangePoints")
      }

      if (nChangePoints > 0) {
        val indexes: Stream[Double] = linspace(0, historySize - 1, nChangePoints)
        changepoints_t = typedDS.zipWithIndex.collect {
          case (x, i) if indexes.contains(i) => x
        }
      }

      if (changepoints_t.nonEmpty)
        changepoints_t = changepoints_t.map(a => (a - start) / t_scale)
      else {
        changepoints_t = Stream.fill(1)(0)
        nChangePoints = 1
      }
    }

    /**
      * Generates a vector of linearly spaced values between a and b (inclusive).
      * The returned vector will have length elements, defaulting to 100.
      */
    def linspace(a: Int, b: Int, length: Int = 100): Stream[Double] = {
      val increment = (b - a) / (length - 1)
      Stream.tabulate(length)(i => a + increment * i)
    }
  }

  private def init_param(events: mutable.Map[String, Stream[Any]], seasonalData: SeasonalData): Parameter = {
    val typedDS = getDoubleSequence(events("ds"))
    val typedT = getDoubleSequence(events("t"))
    val typedYscaled = getDoubleSequence(events("y_scaled"))

    val i0 = typedDS.zipWithIndex.maxBy(a => a._1)._2
    val i1 = typedDS.zipWithIndex.minBy(a => a._1)._2
    val t = typedT(i1) - typedT(i0)
    val k = (typedYscaled(i1) - typedYscaled(i0)) / t
    val m = typedYscaled(i0) - k * typedT(i0)
    Domain.Parameter(k, m, Seq.fill(changepoints_t.length)(0d), 1d, Seq.fill(seasonalData.prior_scales.length)(0d))
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

  def add_regressor(name: String, standardize: Option[Boolean],
                    priorScale: Option[Double] = None, mode: Option[String] = None): Prophet = {
    if (history.nonEmpty) throw new RuntimeException("Regressors must be added prior to model fitting.")
    validate_column_name(name, check_regressors = false)
    val ps = priorScale.getOrElse(seasonalityPriorScale)
    if (ps <= 0) throw new RuntimeException("Prior scale must be >0")

    val md = mode.getOrElse(seasonalityMode)
    if (!Seq("additive", "multiplicative").contains(md))
      throw new RuntimeException("mode must be 'additive' or 'multiplicative'")
    regressors = regressors :+ Domain.Regressor(name, 0, 1, standardize, ps, md)
    this
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


  def add_seasonality(name: String, period: Double, fourier_order: Int,
                      prior_scale: Option[Double] = None, mode: Option[String] = None): Prophet = {

    if (history.nonEmpty)
      throw new RuntimeException("Seasonality must be added prior to model fitting.")
    if (!Seq("daily", "weekly", "yearly").contains(name)) {
      // Allow overwriting built-in seasonalities
      validate_column_name(name, check_seasonalities = false)
    }
    val ps = prior_scale.getOrElse(seasonalityPriorScale)
    if (ps <= 0) throw new RuntimeException("Prior scale must be >0")

    val md = mode.getOrElse(seasonalityMode)
    if (!Seq("additive", "multiplicative").contains(md))
      throw new RuntimeException("mode must be 'additive' or 'multiplicative'")
    seasonalities = seasonalities :+ Seasonality(name, period, fourier_order, ps, md)
    this
  }

  /**
    *
    * @param name                :String
    * @param check_holidays      :Boolean, check if name already used for holiday
    * @param check_seasonalities :Boolean, check if name already used for seasonality
    * @param check_regressors    :Boolean, check if name already used for regressor
    * @return
    */
  private def validate_column_name(name: String, check_holidays: Boolean = true,
                                   check_seasonalities: Boolean = true, check_regressors: Boolean = true): Unit = {

    if (name.contains('_delim_)) throw new RuntimeException("Name cannot contain '_delim_'")
    var reserved_names = Seq("trend", "additive_terms", "daily", "weekly", "yearly", "holidays",
      "zeros", "extra_regressors_additive", "yhat", "extra_regressors_multiplicative",
      "multiplicative_terms", "ds", "y", "cap", "floor", "y_scaled", "cap_scaled")
    val rn_l = reserved_names.map(a => a + "_lower")
    val rn_u = reserved_names.map(a => a + "_upper")
    reserved_names = reserved_names ++ rn_l ++ rn_u
    if (reserved_names.contains(name)) {
      throw new RuntimeException(s"Name  $name  is reserved")
    }
    if (check_seasonalities && seasonalities.map(s => s.name).contains(name))
      throw new RuntimeException(s"Name $name is already used for a seasonality")

    if (check_regressors && regressors.map(s => s.name).contains(name))
      throw new RuntimeException(s"Name $name is already used for a regressor")
  }

  /*
      Validates the inputs to Prophet
   */
  private def validate_inputs() {
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
}

object Prophet {

  def apply(growth: String = "linear",
            changePoints: Stream[Double] = Stream.empty,
            nChangePoints: Int = 25,
            changePointRange: Double = 0.8,
            yearlySeasonality: Any = "auto",
            weeklySeasonality: Any = "auto",
            dailySeasonality: Any = "auto",
            seasonalityMode: String = "additive",
            seasonalityPriorScale: Double = 10.0,
            changePointPriorScale: Double = 0.05,
            mcmcSamples: Int = 0,
            intervalWidth: Double = 0.80, // Number of regressors
            uncertaintySamples: Int = 1000 // Capacities for logistic trend
           ): Prophet =
    new Prophet(growth, changePoints, nChangePoints, changePointRange,
      yearlySeasonality, weeklySeasonality, dailySeasonality
      , seasonalityMode, seasonalityPriorScale, changePointPriorScale
      , mcmcSamples, intervalWidth, uncertaintySamples)

  private def makeAllSeasonalityFeatures(events: mutable.Map[String, Stream[Any]],
                                         seasonalities: Seq[Seasonality], regressors: Seq[Regressor]): SeasonalData = {

    var seasonalFeatures: Stream[Stream[Double]] = Stream.empty
    var prior_scales: Stream[Double] = Stream.empty
    var s_a: Stream[Any] = Stream.empty
    var s_m: Stream[Any] = Stream.empty

    val typedDS = getDoubleSequence(events("ds"))

    seasonalities.foreach(seasonality => {
      val currentFeaturesMap = makeSeasonalityFeatures(typedDS, seasonality)
      seasonalFeatures = seasonalFeatures #::: currentFeaturesMap.values.toStream
      prior_scales = prior_scales #::: Stream.fill(currentFeaturesMap.size)(seasonality.priorScale)
      if (seasonality.mode == "additive") {
        s_a = s_a #::: Stream.fill(currentFeaturesMap.size)(1d)
        s_m = s_m #::: Stream.fill(currentFeaturesMap.size)(0d)
      } else {
        s_a = s_a #::: Stream.fill(currentFeaturesMap.size)(0d)
        s_m = s_m #::: Stream.fill(currentFeaturesMap.size)(1d)
      }
    })

    regressors.foreach(regressor => {
      val currentFeaturesMap = Map(regressor.name -> getDoubleSequence(events(regressor.name)))
      seasonalFeatures = seasonalFeatures #::: currentFeaturesMap.values.toStream
      prior_scales = prior_scales #::: Stream(regressor.priorScale)
      if (regressor.mode == "additive") {
        s_a = s_a #::: Stream.fill(currentFeaturesMap.size)(1d)
        s_m = s_m #::: Stream.fill(currentFeaturesMap.size)(0d)
      } else {
        s_a = s_a #::: Stream.fill(currentFeaturesMap.size)(0d)
        s_m = s_m #::: Stream.fill(currentFeaturesMap.size)(1d)
      }
    })

    if (seasonalFeatures.isEmpty) {
      seasonalFeatures.append(Stream.fill(events("ds").length)(0d))
      prior_scales = prior_scales #::: Stream(1d)
      s_a = s_a #::: Stream.fill(1)(0d)
      s_m = s_m #::: Stream.fill(1)(0d)
    }
    SeasonalData(seasonalFeatures.transpose, prior_scales, getDoubleSequence(s_a), getDoubleSequence(s_m))
  }

  /**
    *
    * @param t           : Stream containing epochs
    * @param seasonality :  Seasonality
    * @return
    */
  private def makeSeasonalityFeatures(t: Stream[Double],
                                      seasonality: Seasonality): mutable.Map[String, Stream[Double]] = {
    val events: mutable.Map[String, Stream[Double]] = mutable.Map.empty
    val features = fourier_series(t, seasonality.period, seasonality.fourierOrder)
    features.zipWithIndex.foreach(a => {
      events(seasonality.name + "_delim_" + a._2) = a._1
    })
    events
  }

  /**
    * Provides Fourier series components with the specified frequency
    * and order.
    *
    * @param t            : Stream containing epochs
    * @param period       :  Number of days of the period.
    * @param series_order : Number of components.
    */
  private def fourier_series(t: Stream[Double], period: Double, series_order: Int) = {

    (0 to (2 * (series_order - 1)) + 1).map(i => {
      t.map(a => a / (3600 * 24 * 1000)).map(a => {
        if (i % 2 == 0) {
          math.sin((2.0 * (i / 2 + 1) * math.Pi * a) / period)
        } else {
          math.cos((2.0 * (i / 2 + 1) * math.Pi * a) / period)
        }
      })
    })
  }


  private def getDoubleSequence(any: Stream[Any]): Stream[Double] = {
    any.map(a => a.asInstanceOf[Double])
  }

  // Needs dummy because of scala stan bug
  case class Newton() extends OptimizeAlgorithm("newton") {
    def arguments: Seq[String] = build(("DUMMY", "DUMMY", "DUMMY"))
  }

}