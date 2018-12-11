import com.cibo.scalastan.RunMethod.Optimize
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.prophet4s.domain.Domain.Seasonality
import org.prophet4s.model.Prophet
import org.prophet4s.model.Prophet.Newton
import org.scalactic.{Equality, TolerantNumerics}
import org.scalatest.FunSuite

import scala.collection.mutable

/**
  * User: ratnesh
  * Date: 05/12/18
  * Time: 11:12 AM
  */
class TestProphet extends FunSuite {
  val dummy = Prophet()


  private val path: String = getClass.getResource("/data/data.csv").getPath
  private val source = com.cibo.scalastan.data.CsvDataSource.fromFile(path)
  private val data: Seq[Double] = source.read(dummy.y, "y")
  private val ds: Seq[Double] = source.read(dummy.t, "ds")
  private val length: Int = data.length


  private val path2: String = getClass.getResource("/data/data2.csv").getPath
  private val source2 = com.cibo.scalastan.data.CsvDataSource.fromFile(path2)
  private val data2: Seq[Double] = source2.read(dummy.y, "y")
  private val ds2: Seq[Double] = source2.read(dummy.t, "ds")
  private val length2: Int = data2.length

  test("fitPredict") {
    val history: Map[String, INDArray] = Map("y" -> Nd4j.create(data.slice(0, length / 2).toArray), "ds" -> Nd4j.create(ds.slice(0, length / 2).toArray))
    val future: Map[String, INDArray] = Map("ds" -> Nd4j.create(ds.slice(length / 2, length).toArray))
    val model = Prophet()
    model.fit(history)
    model.predict(future)
  }

  test("fitPredictNoSeason") {
    val history: Map[String, INDArray] = Map("y" -> Nd4j.create(data.slice(0, length / 2).toArray), "ds" -> Nd4j.create(ds.slice(0, length / 2).toArray))
    val future: Map[String, INDArray] = Map("ds" -> Nd4j.create(ds.slice(length / 2, length).toArray))
    val model = Prophet(dailySeasonality = false, weeklySeasonality = false, yearlySeasonality = false)
    model.fit(history)
    model.predict(future)
  }

  test("fitPredictNoChangePoints") {
    val history: Map[String, INDArray] = Map("y" -> Nd4j.create(data.slice(0, length / 2).toArray), "ds" -> Nd4j.create(ds.slice(0, length / 2).toArray))
    val future: Map[String, INDArray] = Map("ds" -> Nd4j.create(ds.slice(length / 2, length).toArray))
    val model = Prophet(nChangePoints = 0)
    model.fit(history)
    model.predict(future)
  }

  test("fitChangePointNotInHistory") {
    val history: Map[String, INDArray] = Map("y" -> Nd4j.create(data.slice(0, length / 2).toArray), "ds" -> Nd4j.create(ds.slice(0, length / 2).toArray))
    val future: Map[String, INDArray] = Map("ds" -> Nd4j.create(ds.slice(length / 2, length).toArray))
    val model = Prophet(changePoints = Some(Nd4j.create(Array.fill(1)(1200000000000d))))

    val thrown = intercept[UnsupportedOperationException] {
      model.fit(history)
      model.predict(future)
    }
    assert(thrown.getMessage === "Changepoints must fall within training data")
  }


  test("fitPredictDuplicates") {
    val train1 = data.slice(0, length / 2)
    val train2 = data.slice(0, length / 2).map(a => a + 10)
    val dsDup = ds.slice(0, length / 2) ++ ds.slice(0, length / 2)
    val train = train1 ++ train2

    val history: Map[String, INDArray] = Map("y" -> Nd4j.create(train.toArray), "ds" -> Nd4j.create(dsDup.toArray))
    val future: Map[String, INDArray] = Map("ds" -> Nd4j.create(ds.slice(length / 2, length).toArray))
    val model = Prophet()
    model.fit(history)
    model.predict(future)
  }

  test("fitPredictConstantHistory") {
    val train = data.slice(0, length / 2).map(_ => 20d)
    val history: Map[String, INDArray] = Map("y" -> Nd4j.create(train.toArray), "ds" -> Nd4j.create(ds.slice(0, length / 2).toArray))
    val future: Map[String, INDArray] = Map("ds" -> Nd4j.create(ds.slice(length / 2, length).toArray))
    val model = Prophet()
    model.fit(history)
    val predicted = model.predict(future)
    assert(predicted("yhat").getDouble(predicted("yhat").length() - 1) === 20d)
  }

  test("fitPredictConstantHistoryZero") {
    val train = data.slice(0, length / 2).map(_ => 0d)
    val history: Map[String, INDArray] = Map("y" -> Nd4j.create(train.toArray), "ds" -> Nd4j.create(ds.slice(0, length / 2).toArray))
    val future: Map[String, INDArray] = Map("ds" -> Nd4j.create(ds.slice(length / 2, length).toArray))
    val model = Prophet()
    model.fit(history)
    val predicted = model.predict(future)
    assert(predicted("yhat").getDouble(predicted("yhat").length() - 1) === 0d)
  }

  test("regularize") {
    val model = Prophet()
    val events = model.prepare(events = mutable.Map("y" -> Nd4j.create(data.slice(0, length / 2).toArray), "ds" -> Nd4j.create(ds.slice(0, length / 2).toArray)), initializeScales = true)

    assert(events.contains("t"))
    assert(Nd4j.max(events("t")).getDouble(0) == 1d)
    assert(Nd4j.min(events("t")).getDouble(0) == 0d)

    assert(events.contains("y_scaled"))
    assert(Nd4j.max(events("y_scaled")).getDouble(0) == 1d)
  }

  test("logisticFloor") {
    val model = Prophet(growth = "logistic")
    val history = Map(
      "y" -> Nd4j.create(data.slice(0, length / 2).toArray),
      "ds" -> Nd4j.create(ds.slice(0, length / 2).toArray),
      "cap" -> Nd4j.create(Array.fill(length / 2)(80d)),
      "floor" -> Nd4j.create(Array.fill(length / 2)(10d))
    )
    model.fit(events = history, method = Some(Optimize(iter = 10000, algorithm = Newton())))

    assert(model.logisticFloor)
    assert(model.history.contains("floor"))
    implicit val doubleEq: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(1e-1f)
    assert(model.history("y_scaled").getDouble(0) === 1d)


    val future: Map[String, INDArray] = Map(
      "ds" -> Nd4j.create(ds.slice(length / 2, length).toArray),
      "cap" -> Nd4j.create(Array.fill(length / 2)(80d)),
      "floor" -> Nd4j.create(Array.fill(length / 2)(10d))
    )
    val predicted = model.predict(future)

    val modelS = Prophet(growth = "logistic")
    val historyS = Map(
      "y" -> Nd4j.create(data.slice(0, length / 2).map(a => a + 10d).toArray),
      "ds" -> Nd4j.create(ds.slice(0, length / 2).toArray),
      "cap" -> Nd4j.create(Array.fill(length / 2)(80d).map(a => a + 10d)),
      "floor" -> Nd4j.create(Array.fill(length / 2)(10d).map(a => a + 10d))
    )
    modelS.fit(events = historyS, method = Some(Optimize(iter = 10000, algorithm = Newton())))

    val futureS: Map[String, INDArray] = Map(
      "ds" -> Nd4j.create(ds.slice(length / 2, length).toArray),
      "cap" -> Nd4j.create(Array.fill(length / 2)(80d).map(a => a + 10d)),
      "floor" -> Nd4j.create(Array.fill(length / 2)(10d).map(a => a + 10d))
    )
    val predictedS = modelS.predict(futureS)
    // Check for approximate shift invariance
    predicted("yhat").sub(predictedS("yhat").sub(10)).data().asDouble().foreach(a => assert(a === 0d))
  }

  test("getChangePoints") {
    val history: mutable.Map[String, INDArray] = mutable.Map("y" -> Nd4j.create(data.slice(0, length / 2).toArray), "ds" -> Nd4j.create(ds.slice(0, length / 2).toArray))
    val model = Prophet()
    model.history = model.prepare(events = history, initializeScales = true)
    model.setChangePoints()
    val changePoints = model.changePoints.get
    assert(changePoints.length() == model.nChangePoints)
    assert(Nd4j.min(changePoints).getDouble(0) > 0)
    assert(Nd4j.max(changePoints).getDouble(0) <= history("t").getDouble(Math.ceil(0.8 * history("t").length()).intValue()))
  }

  test("setChangePointRange") {

    val history: mutable.Map[String, INDArray] = mutable.Map("y" -> Nd4j.create(data.slice(0, length / 2).toArray), "ds" -> Nd4j.create(ds.slice(0, length / 2).toArray))
    val model = Prophet(changePointRange = 0.4)
    model.history = model.prepare(events = history, initializeScales = true)
    model.setChangePoints()
    val changePoints = model.changePoints.get
    assert(changePoints.length() == model.nChangePoints)
    assert(Nd4j.min(changePoints).getDouble(0) > 0)
    assert(Nd4j.max(changePoints).getDouble(0) <= history("t").getDouble(Math.ceil(0.4 * history("t").length()).intValue()))

    val thrown1 = intercept[UnsupportedOperationException] {
      Prophet(changePointRange = -0.1)
    }
    assert(thrown1.getMessage === "Parameter 'changePointRange' must be in [0, 1]")

    val thrown2 = intercept[UnsupportedOperationException] {
      Prophet(changePointRange = 2)
    }
    assert(thrown2.getMessage === "Parameter 'changePointRange' must be in [0, 1]")
  }

  test("getZeroChangePoints") {
    val history: mutable.Map[String, INDArray] = mutable.Map("y" -> Nd4j.create(data.slice(0, length / 2).toArray), "ds" -> Nd4j.create(ds.slice(0, length / 2).toArray))
    val model = Prophet(nChangePoints = 0)
    model.history = model.prepare(events = history, initializeScales = true)
    model.setChangePoints()
    val changePoints = model.changePoints.get
    assert(changePoints.length() == model.nChangePoints)
    assert(changePoints.length() == 1)
    assert(changePoints.getDouble(0) == 0d)
  }

  test("overridenChangePoints") {
    val history: mutable.Map[String, INDArray] = mutable.Map("y" -> Nd4j.create(data.slice(0, 20).toArray), "ds" -> Nd4j.create(ds.slice(0, 20).toArray))
    val model = Prophet()
    model.history = model.prepare(events = history, initializeScales = true)
    model.setChangePoints()
    val changePoints = model.changePoints.get
    assert(changePoints.length() == model.nChangePoints)
    assert(changePoints.length() == 15)
  }

  test("fourierSeriesWeekly") {
    val mat = Prophet.fourierSeries(t = Nd4j.create(ds.toArray), period = 7, series_order = 3)
    // These are from the R forecast package directly.
    val trueValues = Nd4j.create(Array(0.7818315, 0.6234898, 0.9749279, -0.2225209, 0.4338837, -0.9009689))
    implicit val doubleEq: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(1e-4f)
    assert(Nd4j.sum(mat.getRow(0).sub(trueValues)).getDouble(0) === 0.0)
  }

  test("fourierSeriesYearly") {
    val mat = Prophet.fourierSeries(t = Nd4j.create(ds.toArray), period = 365.25, series_order = 3)
    // These are from the R forecast package directly.
    val trueValues = Nd4j.create(Array(0.7006152, -0.7135393, -0.9998330, 0.01827656, 0.7262249, 0.6874572))
    implicit val doubleEq: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(1e-4f)
    assert(Nd4j.sum(mat.getRow(0).sub(trueValues)).getDouble(0) === 0.0)
  }

  test("growthInit") {
    val model = Prophet(growth = "logistic")
    var history: mutable.Map[String, INDArray] = mutable.Map("y" -> Nd4j.create(data.slice(0, 468).toArray),
      "ds" -> Nd4j.create(ds.slice(0, 468).toArray),
      "cap" -> Nd4j.create(Array.fill(468)(data.slice(0, 468).max)))

    history = model.prepare(history, initializeScales = true)
    val parameter = Prophet.initParam(history, growth = "linear", model.nChangePoints, 1)
    implicit val doubleEq: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(1e-4f)

    assert(parameter.k.getDouble(0) === 0.3055671d)
    assert(parameter.m.getDouble(0) === 0.5307511d)

    val loParameter = Prophet.initParam(history, growth = "logistic", model.nChangePoints, 1)

    assert(loParameter.k.getDouble(0) === 1.507925d)
    assert(loParameter.m.getDouble(0) === -0.08167497d)
  }

  test("piecewiseLinear") {
    val tSeq = Seq.tabulate(11)(i => i.toDouble)
    val t = Nd4j.create(tSeq.toArray[Double])
    val deltas = Nd4j.create(Array(0.5d))
    val changePoints = Nd4j.create(Array(5d))
    val mat = Prophet.piecewiseLinear(t, deltas = deltas, k = 1d, m = 0d, changePoints = changePoints)
    val trueArray = Array(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 9.5, 11.0, 12.5)
    val trueValues = Nd4j.create(trueArray)
    implicit val doubleEq: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(1e-4f)
    assert(Nd4j.sum(mat.getRow(0).sub(trueValues)).getDouble(0) === 0.0)

    val t1 = Nd4j.create(tSeq.drop(8).toArray[Double])
    val mat1 = Prophet.piecewiseLinear(t1, deltas = deltas, k = 1d, m = 0d, changePoints = changePoints)
    val trueValues1 = Nd4j.create(trueArray.drop(8))
    assert(Nd4j.sum(mat1.getRow(0).sub(trueValues1)).getDouble(0) === 0.0)
  }

  test("piecewiseLogistic") {
    val tSeq = Seq.tabulate(11)(i => i.toDouble)
    val t = Nd4j.create(tSeq.toArray[Double])
    val capArray = Array.fill(11)(10d)
    val cap = Nd4j.create(capArray)

    val deltas = Nd4j.create(Array(0.5d))
    val changePoints = Nd4j.create(Array(5d))
    val mat = Prophet.piecewiseLogistic(t, cap = cap, deltas = deltas, k = 1d, m = 0d, changePoints = changePoints)
    val trueArray = Array(5.000000, 7.310586, 8.807971, 9.525741, 9.820138,
      9.933071, 9.984988, 9.996646, 9.999252, 9.999833,
      9.999963)
    val trueValues = Nd4j.create(trueArray)
    implicit val doubleEq: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(1e-4f)
    assert(Nd4j.sum(mat.getRow(0).sub(trueValues)).getDouble(0) === 0.0)

    val t1 = Nd4j.create(tSeq.drop(8).toArray[Double])
    val cap1 = Nd4j.create(capArray.drop(8))
    val mat1 = Prophet.piecewiseLogistic(t1, cap = cap1, deltas = deltas, k = 1d, m = 0d, changePoints = changePoints)
    val trueValues1 = Nd4j.create(trueArray.drop(8))
    assert(Nd4j.sum(mat1.getRow(0).sub(trueValues1)).getDouble(0) === 0.0)
  }

  test("autoWeeklySeasonality") {
    // Should be enabled
    var history: Map[String, INDArray] = Map("y" -> Nd4j.create(data.slice(0, 15).toArray), "ds" -> Nd4j.create(ds.slice(0, 15).toArray))
    var model = Prophet()
    assert(model.weeklySeasonality == "auto")
    model.fit(history)
    assert(model.seasonalities.map(a => a.name).contains("weekly"))
    assert(model.seasonalities.filter(a => a.name == "weekly") == mutable.Seq(Seasonality("weekly", 7, 3, 10)))

    //Should be disabled
    history = Map("y" -> Nd4j.create(data.slice(0, 9).toArray), "ds" -> Nd4j.create(ds.slice(0, 9).toArray))
    model = Prophet()
    model.fit(history)
    assert(!model.seasonalities.map(a => a.name).contains("weekly"))

    model = Prophet(weeklySeasonality = true)
    model.fit(history)
    assert(model.seasonalities.map(a => a.name).contains("weekly"))

    // Should be False due to weekly spacing
    history = Map("y" -> Nd4j.create(data.zipWithIndex.filter(a => a._2 % 7 == 0).map(a => a._1).toArray),
      "ds" -> Nd4j.create(ds.zipWithIndex.filter(a => a._2 % 7 == 0).map(a => a._1).toArray))
    model = Prophet()
    model.fit(history)
    assert(!model.seasonalities.map(a => a.name).contains("weekly"))

    history = Map("y" -> Nd4j.create(data.toArray), "ds" -> Nd4j.create(ds.toArray))
    model = Prophet(weeklySeasonality = 2, seasonalityPriorScale = 3d)
    model.fit(history)
    assert(model.seasonalities.filter(a => a.name == "weekly") == mutable.Seq(Seasonality("weekly", 7, 2, 3)))
  }

  test("autoYearlySeasonality") {
    // Should be enabled
    var history: Map[String, INDArray] = Map("y" -> Nd4j.create(data.toArray), "ds" -> Nd4j.create(ds.toArray))
    var model = Prophet()
    assert(model.yearlySeasonality == "auto")
    model.fit(history)
    assert(model.seasonalities.map(a => a.name).contains("yearly"))
    assert(model.seasonalities.filter(a => a.name == "yearly") == mutable.Seq(Seasonality("yearly", 365.25, 10, 10)))

    //Should be disabled
    history = Map("y" -> Nd4j.create(data.slice(0, 240).toArray), "ds" -> Nd4j.create(ds.slice(0, 240).toArray))
    model = Prophet()
    model.fit(history)
    assert(!model.seasonalities.map(a => a.name).contains("yearly"))

    model = Prophet(yearlySeasonality = true)
    model.fit(history)
    assert(model.seasonalities.map(a => a.name).contains("yearly"))

    history = Map("y" -> Nd4j.create(data.toArray), "ds" -> Nd4j.create(ds.toArray))
    model = Prophet(yearlySeasonality = 7, seasonalityPriorScale = 3)
    model.fit(history)
    assert(model.seasonalities.filter(a => a.name == "yearly") == mutable.Seq(Seasonality("yearly", 365.25, 7, 3)))
  }

  test("autoDailySeasonality") {
    // Should be enabled
    var history: Map[String, INDArray] = Map("y" -> Nd4j.create(data2.toArray), "ds" -> Nd4j.create(ds2.toArray))
    var model = Prophet()
    assert(model.dailySeasonality == "auto")
    model.fit(history)
    assert(model.seasonalities.map(a => a.name).contains("daily"))
    assert(model.seasonalities.filter(a => a.name == "daily") == mutable.Seq(Seasonality("daily", 1, 4, 10)))

    //Should be disabled
    history = Map("y" -> Nd4j.create(data2.slice(0, 430).toArray), "ds" -> Nd4j.create(ds2.slice(0, 430).toArray))
    model = Prophet()
    model.fit(history)
    assert(!model.seasonalities.map(a => a.name).contains("daily"))

    model = Prophet(dailySeasonality = true)
    model.fit(history)
    assert(model.seasonalities.map(a => a.name).contains("daily"))

    history = Map("y" -> Nd4j.create(data2.toArray), "ds" -> Nd4j.create(ds2.toArray))
    model = Prophet(dailySeasonality = 7, seasonalityPriorScale = 3)
    model.fit(history)
    assert(model.seasonalities.filter(a => a.name == "daily") == mutable.Seq(Seasonality("daily", 1, 7, 3)))

    history = Map("y" -> Nd4j.create(data.toArray), "ds" -> Nd4j.create(ds.toArray))
    model = Prophet()
    model.fit(history)
    assert(!model.seasonalities.map(a => a.name).contains("daily"))
  }

  test("addCustomSeasonality") {
    var model = Prophet()
    model.addSeasonality("monthly", 30, 5, Some(2))
    assert(model.seasonalities.filter(a => a.name == "monthly") == mutable.Seq(Seasonality("monthly", 30, 5, 2)))

    val thrown1 = intercept[UnsupportedOperationException] {
      model.addSeasonality("trend", 30, 5, Some(2))
    }
    assert(thrown1.getMessage === "Name  trend  is reserved")


    model.addSeasonality("weekly", 7, 2, Some(3))

    model = Prophet(seasonalityMode = "multiplicative")
    model.addSeasonality("monthly", 30, 5, Some(2), Some("additive"))
    val history: Map[String, INDArray] = Map("y" -> Nd4j.create(data.toArray), "ds" -> Nd4j.create(ds.toArray))
    model.fit(history)
    assert(model.seasonalities.filter(a => a.name == "monthly").map(a => a.mode).contains("additive"))
    assert(model.seasonalities.filter(a => a.name == "weekly").map(a => a.mode).contains("multiplicative"))
  }
}
