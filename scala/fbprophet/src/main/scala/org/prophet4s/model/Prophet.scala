package org.prophet4s.model

import java.util.concurrent.TimeUnit

import com.cibo.scalastan.RunMethod.OptimizeAlgorithm
import com.cibo.scalastan.{RunMethod, StanResults}
import org.apache.commons.math3.distribution.{LaplaceDistribution, NormalDistribution, PoissonDistribution}
import org.nd4j.linalg.api.buffer.DataBuffer.Type
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms.{abs, exp}
import org.prophet4s.domain.Domain
import org.prophet4s.domain.Domain.{Parameter, Regressor, SeasonalData, Seasonality}
import org.prophet4s.model.Prophet._
import org.prophet4s.stan.ProphetModel

import scala.collection.mutable

/**
  * Prophet forecaster.
  *
  * @param growth                                    : 'linear' or 'logistic' to specify a linear or logistic trend.
  * @param changepoints                              : List of epochs at which to include potential changepoints.
  *                                                  If not specified, potential changepoints are selected automatically.
  * @param nChangepoints                             : Number of potential changepoints to include. Not used
  *                                                  if input `changepoints` is supplied. If `changepoints` is not supplied,
  *                                                  then nChangePoints potential changepoints are selected uniformly from
  *                                                  the first `changePointRange` proportion of the history.
  * @param changePointRange                          : Proportion of history in which trend changepoints will
  *                                                  be estimated. Defaults to 0.8 for the first 80%. Not used if
  *                                                  `changepoints` is specified.
  *                                                  Not used if input `changepoints` is supplied.
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
  * @param intervalWidth : Float, width of the uncertainty intervals provided
  *                                                  for the forecast. If mcmc_samples=0, this will be only the uncertainty
  *                                                  in the trend using the MAP estimate of the extrapolated generative
  *                              model. If mcmc.samples>0, this will be integrated over all model
  * @param uncertaintySamples                        : Number of simulated draws used to estimate
  *                                                  uncertainty intervals.
  */
class Prophet(growth: String = "linear",
              changepoints: Option[INDArray] = None,
              nChangepoints: Int = 25,
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
             ) extends ProphetModel {
  Nd4j.setDataType(Type.DOUBLE)
  private var changePoints: Option[INDArray] = changepoints
  private var nChangePoints = if (changePoints.isEmpty) nChangepoints else changepoints.get.length()

  validate_inputs()
  private var seasonalities: mutable.Seq[Seasonality] = mutable.Seq.empty
  private var regressors: mutable.Seq[Regressor] = mutable.Seq.empty

  // Set during fitting
  private var history: collection.mutable.Map[String, INDArray] = mutable.Map.empty
  private var start: Double = 0
  private var y_scale: Double = 0
  private var logistic_floor: Boolean = false
  private var t_scale: Double = 0

  private val doubles = new Array[Double](1)
  //Set during fitting
  private var parameter: Parameter = Parameter(create(Array(0d)), create(Array(0d)),
    create(1, 1), create(Array(0d)), create(1, 1))


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

  def fit(events: Map[String, INDArray]): Prophet = {

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
    val nSeasonalFeatures = seasonalData.prior_scales.length
    parameter = init_param(history, growth, changePoints.get.length(), nSeasonalFeatures)
    val yData: Seq[Double] = history("y_scaled").data().asDouble()
    val deltaData: Seq[Double] = parameter.delta.data().asDouble().toSeq
    val betaData: Seq[Double] = parameter.beta.data().asDouble().toSeq
    val seasonalFeatures: Seq[Seq[Double]] = seasonalData.seasonalFeatures.data().asDouble().toSeq.sliding(nSeasonalFeatures, nSeasonalFeatures).toSeq
    val model = super.compile
      .withInitialValue(k, parameter.k.getDouble(0))
      .withInitialValue(m, parameter.m.getDouble(0))
      .withInitialValue(delta, parameter.delta.data().asDouble().toSeq)
      .withInitialValue(beta, parameter.beta.data().asDouble().toSeq)
      .withInitialValue(sigma_obs, parameter.sigma_obs.getDouble(0))
      .withData(cap, history("cap_scaled").data().asDouble().toSeq)
      .withData(T, history("ds").length)
      .withData(K, nSeasonalFeatures)
      .withData(S, changePoints.get.length())
      .withData(y, yData)
      .withData(t, history("t").data().asDouble().toSeq)
      .withData(t_change, changePoints.get.data().asDouble().toSeq)
      .withData(X, seasonalFeatures)
      .withData(sigmas, seasonalData.prior_scales)
      .withData(tau, changePointPriorScale)
      .withData(trend_indicator, if (growth == "linear") 0 else 1)
      .withData(s_a, seasonalData.s_a.data().asDouble().toSeq)
      .withData(s_m, seasonalData.s_m.data().asDouble().toSeq)

    if (growth == "linear" && max(history("y")).getDouble(0) == min(history("y_scaled")).getDouble(0)) {
      parameter = Parameter(parameter.k, parameter.m, parameter.delta, create(Array(Math.pow(10, -9))), parameter.beta)
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
      val _k = create(results.samples(k).apply(results.bestChain).toArray[Double])
      val _m = create(results.samples(m).apply(results.bestChain).toArray[Double])
      val _delta = create(results.samples(delta).apply(results.bestChain).map(a => a.toArray[Double]).toArray)
      val _sigma_obs = create(results.samples(sigma_obs).apply(results.bestChain).toArray[Double])
      val _beta = create(results.samples(beta).apply(results.bestChain).map(a => a.toArray[Double]).toArray)
      Parameter(_k, _m, _delta, _sigma_obs, _beta)
    }

    this
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
    val typedDS = history("ds")
    val first = min(typedDS).getDouble(0)
    val last = max(typedDS).getDouble(0)
    var min_dt = 0d
    (0 until typedDS.length() - 1).foreach(i => {
      val diff = typedDS.getDouble(i + 1) - typedDS.getDouble(i)
      if (diff > 0 && diff > min_dt) min_dt = diff
    }
    )

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
    val typedDS = history("ds")
    if (changePoints.nonEmpty) {
      val tooLow = min(changePoints.get).getDouble(0) < min(typedDS).getDouble(0)
      val tooHigh = max(changePoints.get).getDouble(0) < max(typedDS).getDouble(0)
      if (tooHigh || tooLow)
        throw new RuntimeException("Changepoints must fall within training data")
    } else {
      val historySize = Math.floor(typedDS.length * changePointRange).toInt
      if (nChangePoints + 1 > historySize) {
        nChangePoints = historySize - 1
        logger.info(s"nChangePoints greater than number of observations. Using $nChangePoints")
      }

      if (nChangePoints > 0) {
        val indexes: Stream[Int] = linspace(0, historySize - 1, nChangePoints)
        changePoints = scala.Some(zeros(nChangepoints))
        indexes.zipWithIndex.foreach(index => changePoints.get.put(index._2, create(Array.fill(1)(typedDS.getDouble(index._1)))))
      }

      if (changePoints.nonEmpty)
        changePoints = Some(changePoints.get.sub(start).div(t_scale))
      else {
        changePoints = scala.Some(zeros(1))
        nChangePoints = 1
      }
    }

    /**
      * Generates a vector of linearly spaced values between a (exclusive) and b (inclusive).
      * The returned vector will have length elements, defaulting to 100.
      */
    def linspace(a: Int, b: Int, length: Int = 100): Stream[Int] = {
      val increment = (b - a) / (length - 1)
      Stream.tabulate(length + 1)(i => a + increment * i).drop(1)
    }
  }

  /**
    * Predict using the prophet model.
    *
    * @param events : Map, must have keys ds (time in epoch)
    *               the time series. If growth is 'logistic', then
    *               df must also have a key cap that specifies the capacity at
    *               each ds.
    */
  def predict(events: Map[String, INDArray]): Map[String, INDArray] = {
    var toPredict: collection.mutable.Map[String, INDArray] = mutable.Map.empty

    if (events.isEmpty) {
      toPredict ++= history
    } else {
      toPredict ++= events
      toPredict = setup_history(toPredict)
    }
    toPredict("trend") = predict_trend(toPredict)
    toPredict = predict_seasonal_components(toPredict)
    toPredict("yhat") = toPredict("trend").mul(toPredict("multiplicative_terms").add(1)).add(toPredict("additive_terms"))
    toPredict ++= predictUncertainty(toPredict)
    toPredict.toMap
  }

  /**
    *
    * @param events : Map with ds, t,y_scaled, and cap_scaled if logistic growth.
    * @return
    */
  private def predict_trend(events: mutable.Map[String, INDArray]): INDArray = {

    val t: INDArray = events("t")
    val k: Double = mean(parameter.k, 0).getDouble(0)
    val m: Double = mean(parameter.m, 0).getDouble(0)
    val deltas: INDArray = mean(parameter.delta, 0)

    var trend: INDArray = zeros(1, 1)
    if (growth == "linear") {
      trend = piecewise_linear(t, deltas, k, m, changePoints.get)
    }
    else {
      trend = piecewise_logistic(t, events("cap_scaled"), deltas, k, m, changePoints.get)
    }
    trend.mul(y_scale).add(events("floor")).transpose()
  }

  private def predict_seasonal_components(events: mutable.Map[String, INDArray]): mutable.Map[String, INDArray] = {

    val seasonalData: SeasonalData = makeAllSeasonalityFeatures(events, seasonalities, regressors)
    val lower_p = 100 * (1.0 - intervalWidth) / 2
    val upper_p = 100 * (1.0 + intervalWidth) / 2

    val beta_add = parameter.beta.mul(seasonalData.s_a)
    val comp_add = seasonalData.seasonalFeatures.mmul(beta_add.transpose()).mul(y_scale)
    events.put("additive_terms", mean(comp_add, 1))
    if (comp_add.columns() == 1) {
      events.put("additive_terms_lower", comp_add)
      events.put("additive_terms_upper", comp_add)
    } else {
      events.put("additive_terms_lower", comp_add.percentile(lower_p, 1))
      events.put("additive_terms_upper", comp_add.percentile(upper_p, 1))
    }

    val beta_mul = parameter.beta.mul(seasonalData.s_m)
    val comp_mul = seasonalData.seasonalFeatures.mmul(beta_mul.transpose())
    events.put("multiplicative_terms", comp_mul.mean(1))

    if (comp_mul.columns() == 1) {
      events.put("additive_terms_lower", comp_mul)
      events.put("additive_terms_upper", comp_mul)
    } else {
      events.put("multiplicative_terms_lower", comp_mul.percentile(lower_p, 1))
      events.put("multiplicative_terms_upper", comp_mul.percentile(upper_p, 1))
    }
    events
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
  private def setup_history(events: mutable.Map[String, INDArray],
                            initialize_scales: Boolean = false): mutable.Map[String, INDArray] = {

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
    else events("floor") = zeros(events("ds").length)
    val typedY = events("y")

    val floor = events("floor")

    if (growth == "logistic") {
      if (!events.contains("cap"))
        throw new RuntimeException("Capacities must be supplied for logistic growth in column 'cap'")
      val typedCap = events("cap")
      events.put("cap_scaled", typedCap.sub(floor).div(y_scale))
    } else {
      events.put("cap_scaled", zeros(events("ds").length))
    }

    events.put("t", events("ds").sub(start).div(t_scale))
    if (events.contains("y"))
      events.put("y_scaled", typedY.sub(floor).div(y_scale))

    regressors.foreach(
      regressor => {
        events(regressor.name) = events(regressor.name).sub(regressor.mu).div(regressor.std)
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

  private def initializeScales(events: mutable.Map[String, INDArray],
                               initialize_scales: Boolean): Unit = {
    if (!initialize_scales) return
    val typedDS = events("ds")
    var floor: INDArray = create(Array.fill(typedDS.length())(0d))
    val typedY = events("y")

    if (growth == "logistic" && events.contains("floor")) {
      logistic_floor = true
      floor = events("floor")
    }
    y_scale = max(abs(typedY.sub(floor))).getDouble(0)
    if (y_scale == 0d) y_scale = 1d

    start = min(typedDS).getDouble(0)
    t_scale = max(typedDS).getDouble(0) - start

    regressors.foreach(
      regressor => {
        var standarize: Option[Boolean] = regressor.standardize
        val stream = events(regressor.name)
        val n_vals = stream.dup().length
        if (n_vals < 2) standarize = Some(false)
        if (standarize.isEmpty) {
          if (stream.dup == create(Array(1d, 0d))) standarize = Some(false)
          else standarize = Some(true)
        }
        if (standarize.get)
          regressor.mu = stream.mean().getDouble(0)
        regressor.std = stream.std().getDouble(0)
      }
    )
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

  private def predictUncertainty(events: mutable.Map[String, INDArray]): mutable.Map[String, INDArray] = {
    val simValues = samplePosteriorPredictive(events)
    val lowerP = 100 * (1.0 - intervalWidth) / 2
    val upperP = 100 * (1.0 + intervalWidth) / 2

    mutable.Map("yhat_lower" -> simValues("yhat").percentile(lowerP, 0), "yhat_upper" -> simValues("yhat").percentile(upperP, 0),
      "trend_lower" -> simValues("trend").percentile(lowerP, 0), "trend_upper" -> simValues("trend").percentile(upperP, 0))
  }

  /**
    * Prophet posterior\ predictive samples.
    *
    * @param events : Prediction map
    *
    *               Dictionary with posterior predictive samples for the forecast yhat and
    *               for the trend component.
    */
  private def samplePosteriorPredictive(events: mutable.Map[String, INDArray]): Map[String, INDArray] = {

    val nIterations = parameter.k.length()
    val samplePerIter = Math.max(1, Math.ceil(uncertaintySamples / nIterations).toInt)

    // Generate seasonality features once so we can re-use them.
    val seasonalData = makeAllSeasonalityFeatures(events, seasonalities, regressors)
    val yhat = zeros(events("t").length(), nIterations * samplePerIter)
    val trend = zeros(events("t").length(), nIterations * samplePerIter)

    (0 until nIterations).foreach(i => {
      (0 until samplePerIter).foreach(j => {
        val sampleEvents = sampleModel(events, seasonalData, i)
        yhat.putColumn(i * j + j, sampleEvents("yhat"))
        trend.putColumn(i * j + j, sampleEvents("trend"))
      })
    })
    Map("yhat" -> yhat, "trend" -> trend)
  }

  /**
    * Simulate observations from the extrapolated generative model
    *
    * @param events       : Prediction map.
    * @param seasonalData :  seasonal data.
    * @param iteration    : Int sampling iteration to use parameters from.
    * @return
    */
  private def sampleModel(events: mutable.Map[String, INDArray], seasonalData: SeasonalData, iteration: Int): mutable.Map[String, INDArray] = {

    val trend = samplePredictiveTrend(events, iteration)
    val beta = parameter.beta.getRow(iteration)

    val beta_add = beta.mul(seasonalData.s_a)
    val comp_add = seasonalData.seasonalFeatures.mmul(beta_add.transpose()).mul(y_scale)

    val beta_mul = beta.mul(seasonalData.s_m)
    val comp_mul = seasonalData.seasonalFeatures.mmul(beta_mul.transpose())

    val sigma = parameter.sigma_obs.getDouble(iteration)
    val noise = create(new NormalDistribution(0, sigma).sample(events("t").length()).map(a => a * y_scale)).transpose()

    val toReturn: mutable.Map[String, INDArray] = mutable.Map.empty

    toReturn("yhat") = trend.mul(comp_mul.add(1)).add(comp_add).add(noise)
    toReturn("trend") = trend
    toReturn
  }

  /**
    * Simulate the trend using the extrapolated generative model.
    *
    * @param events    : Prediction Map.
    * @param iteration : sampling iteration to use parameters from.
    * @return Nd4J array of simulated trend over events("t").
    */
  private def samplePredictiveTrend(events: mutable.Map[String, INDArray], iteration: Int): INDArray = {
    val k = parameter.k.getDouble(iteration)
    val m = parameter.m.getDouble(iteration)
    val deltas = parameter.delta.getRow(iteration)

    val T = max(events("t")).getDouble(0)
    // New changepoints from a Poisson process with rate S on [1, T]
    var nChanges = 0
    var newChangepoints: Option[INDArray] = None
    var newDeltas: Option[INDArray] = None

    if (T > 1) {
      val S = changepoints.getOrElse(zeros(1, 1)).length()
      val distribution = new PoissonDistribution(S * (T - 1))
      nChanges = distribution.sample()
    }
    if (nChanges > 0) {
      val r = scala.util.Random
      val array: Seq[Double] = (for (i <- 1 to nChanges) yield 1 + r.nextDouble() * (T - 1)).toSeq
      newChangepoints = Some(create(array.toArray[Double]))

      // Get the empirical scale of the deltas, plus epsilon to avoid NaNs.
      val lamda = mean(abs(deltas)).add(1e-8).getDouble(0)

      // Sample deltas
      val distribution = new LaplaceDistribution(0d, lamda)
      newDeltas = Some(create(distribution.sample(nChanges)))
    }

    var mergedChangepoints = changePoints.get
    var mergedDeltas = deltas

    if (newChangepoints.nonEmpty) {
      mergedChangepoints = concat(1, changePoints.get, newChangepoints.get)
      mergedDeltas = concat(1, newDeltas.get, deltas)
    }

    var trend: Option[INDArray] = None
    if (growth == "linear") {
      trend = Some(piecewise_linear(events("t"), mergedDeltas, k, m, mergedChangepoints))
    } else {
      trend = Some(piecewise_logistic(events("t"), events("cap"), mergedDeltas, k, m, mergedChangepoints))
    }

    trend.get.mul(y_scale).add(events("floor")).transpose()
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
            changePoints: Option[INDArray] = None,
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


  /**
    *
    * @param events            : Map with ds, t,y_scaled, and cap_scaled if logistic growth.
    * @param growth            : 'linear' or 'logistic' to specify a linear or logistic trend.
    * @param nChangepoint      : number of changepoints
    * @param nSeasonalFeatures : number of seasonal features
    * @return Initialized Parameter
    */
  private def init_param(events: mutable.Map[String, INDArray], growth: String, nChangepoint: Int, nSeasonalFeatures: Int): Parameter = {
    val ds = events("ds")
    val time = events("t")
    val yScaled = events("y_scaled")


    val i0 = Nd4j.getExecutioner.exec(new IMin(ds), 0).data().asInt()(0)
    val i1 = ds.argMax().data().asInt()(0)
    val t = time.getDouble(i1) - time.getDouble(i0)

    if (growth == "linear") {
      val k = (yScaled.getDouble(i1) - yScaled.getDouble(i0)) / t
      val m = yScaled.getDouble(i0) - k * time.getDouble(i0)
      Domain.Parameter(create(Array(k)), create(Array(m)), create(Array.fill(nChangepoint)(0d)), create(Array(1d)), create(Array.fill(nSeasonalFeatures)(0d)))
    } else {
      val capScaled = events("cap_scaled")
      val c0 = capScaled.getDouble(i0)
      val c1 = capScaled.getDouble(i1)
      val y0 = Math.max(0.01 * c0, Math.min(0.99 * c0, yScaled.getDouble(i0)))
      val y1 = Math.max(0.01 * c1, Math.min(0.99 * c1, yScaled.getDouble(i1)))

      var r0 = c0 / y0
      val r1 = c1 / y1

      if (Math.abs(r0 - r1) <= .01) r0 = 1.05 * r0

      val l0 = Math.log(r0 - 1)
      val l1 = Math.log(r1 - 1)

      val m = (l0 * t) / (l0 - l1)
      val k = (l0 - l1) / t
      Domain.Parameter(create(Array(k)), create(Array(m)), create(Array.fill(nChangepoint)(0d)), create(Array(1d)), create(Array.fill(nSeasonalFeatures)(0d)))
    }
  }

  private def getDoubleSequence(any: Stream[Any]): Stream[Double] = {
    any.map(a => a.asInstanceOf[Double])
  }

  /**
    *
    * @param t            : Stream containing epochs
    * @param changePoints : List of epochs at which to include potential changepoints.
    * @return Stream containing linear prediction
    */
  private def piecewise_linear(t: INDArray, deltas: INDArray, k: Double, m: Double, changePoints: INDArray): INDArray = {

    // Intercept changes
    val gammas = changePoints.mul(deltas).mul(-1)
    val k_t = create(Array.fill(t.length())(1d)).mul(k)
    val m_t = create(Array.fill(t.length())(1d)).mul(m)

    (0 until changePoints.length()).foreach(s => {
      (0 until t.length()).foreach(t1 => {
        if (t.getDouble(t1) > changePoints.getDouble(s)) {
          k_t.put(t1, k_t.getColumn(t1).add(deltas.getColumn(s)))
          m_t.put(t1, m_t.getColumn(t1).add(gammas.getColumn(s)))
        }
      })
    })
    k_t.mul(t).add(m_t)
  }

  /**
    *
    * @param t            : Stream containing epochs
    * @param changePoints : List of epochs at which to include potential changepoints.
    * @param cap          : logistic cap stream
    * @return
    */
  private def piecewise_logistic(t: INDArray, cap: INDArray, deltas: INDArray, k: Double, m: Double, changePoints: INDArray): INDArray = {

    // Intercept changes
    val kCumulative = concat(0, create(Array.fill(1)(k)), deltas.cumsum(0).add(k))
    val gammas = create(Array.fill(changePoints.length)(0d))

    (0 until changePoints.length()).foreach(s => {
      val value = (changePoints.getDouble(s) - m - sum(gammas, 0).getDouble(0)) *
        (1 - (kCumulative.getDouble(s) / kCumulative.getDouble(s + 1)))
      gammas.put(s, create(Array(x = value)))
    })
    val k_t = create(Array.fill(t.length())(1d)).mul(k)
    val m_t = create(Array.fill(t.length())(1d)).mul(m)

    (0 until changePoints.length()).foreach(s => {
      (0 until t.length()).foreach(t1 => {
        if (t.getDouble(t1) > changePoints.getDouble(s)) {
          k_t.put(t1, k_t.getColumn(t1).add(deltas.getColumn(s)))
          m_t.put(t1, m_t.getColumn(t1).add(gammas.getColumn(s)))
        }
      })
    })
    cap.div(exp(k_t.mul(t).add(m_t).mul(-1)))
  }

  private def makeAllSeasonalityFeatures(events: mutable.Map[String, INDArray],
                                         seasonalities: Seq[Seasonality], regressors: Seq[Regressor]): SeasonalData = {
    val typedDS = events("ds")
    var seasonalFeatures: Option[INDArray] = None
    var prior_scales: Stream[Double] = Stream.empty
    var s_a: Stream[Double] = Stream.empty
    var s_m: Stream[Double] = Stream.empty


    seasonalities.foreach(seasonality => {

      val currentFeatures: INDArray = makeSeasonalityFeatures(events("ds"), seasonality)
      if (seasonalFeatures.isEmpty) seasonalFeatures = Some(currentFeatures)
      else seasonalFeatures = Some(concat(1, seasonalFeatures.get, currentFeatures))
      prior_scales = prior_scales #::: Stream.fill(currentFeatures.size(1))(seasonality.priorScale)
      if (seasonality.mode == "additive") {
        s_a = s_a #::: Stream.fill(currentFeatures.size(1))(1d)
        s_m = s_m #::: Stream.fill(currentFeatures.size(1))(0d)
      } else {
        s_a = s_a #::: Stream.fill(currentFeatures.size(1))(0d)
        s_m = s_m #::: Stream.fill(currentFeatures.size(1))(1d)
      }
    })

    regressors.foreach(regressor => {
      val currentFeatures = events(regressor.name)
      if (seasonalFeatures.isEmpty) seasonalFeatures = Some(currentFeatures)
      else seasonalFeatures = Some(concat(1, seasonalFeatures.get, currentFeatures))
      prior_scales = prior_scales #::: Stream(regressor.priorScale)
      if (regressor.mode == "additive") {
        s_a = s_a #::: Stream.fill(currentFeatures.size(1))(1d)
        s_m = s_m #::: Stream.fill(currentFeatures.size(1))(0d)
      } else {
        s_a = s_a #::: Stream.fill(currentFeatures.size(1))(0d)
        s_m = s_m #::: Stream.fill(currentFeatures.size(1))(1d)
      }
    })

    if (seasonalFeatures.isEmpty) {
      seasonalFeatures = Some(create(Array.fill(events("ds").length)(0d)))
      prior_scales = prior_scales #::: Stream(1d)
      s_a = s_a #::: Stream.fill(1)(0d)
      s_m = s_m #::: Stream.fill(1)(0d)
    }
    SeasonalData(seasonalFeatures.get, prior_scales, create(s_a.toArray[Double]), create(s_m.toArray[Double]))
  }

  /**
    *
    * @param t           : Stream containing epochs
    * @param seasonality :  Seasonality
    * @return
    */
  private def makeSeasonalityFeatures(t: INDArray,
                                      seasonality: Seasonality): INDArray = {
    fourier_series(t, seasonality.period, seasonality.fourierOrder)
  }

  /**
    * Provides Fourier series components with the specified frequency
    * and order.
    *
    * @param t            : Stream containing epochs
    * @param period       :  Number of days of the period.
    * @param series_order : Number of components.
    */
  private def fourier_series(t: INDArray, period: Double, series_order: Int): INDArray = {
    val toReturn = zeros(t.length(), 2 * series_order)

    //    sin(t.div(3600 * 24 * 1000).mul(2*(i/2+1)).mul(math.Pi).div(period)
    (0 until t.length()).map(t1 => {
      (0 to (2 * (series_order - 1)) + 1).map(i => {
        if (i % 2 == 0) {
          toReturn.put(t1, i, math.sin(((t.getDouble(t1) / (3600 * 24 * 1000)) * (2 * (i / 2 + 1)) * math.Pi) / period))
        } else {
          toReturn.put(t1, i, math.cos(((t.getDouble(t1) / (3600 * 24 * 1000)) * (2 * (i / 2 + 1)) * math.Pi) / period))
        }
      })
    })
    toReturn
  }

  // Needs dummy because of scala stan bug
  case class Newton() extends OptimizeAlgorithm("newton") {
    def arguments: Seq[String] = build(("DUMMY", "DUMMY", "DUMMY"))
  }

}