import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.prophet4s.model.Prophet
import org.scalatest.FunSuite

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


  test("Prophet.fitPredict") {
    val history: Map[String, INDArray] = Map("y" -> Nd4j.create(data.slice(0, length / 2).toArray), "ds" -> Nd4j.create(ds.slice(0, length / 2).toArray))
    val future: Map[String, INDArray] = Map("y" -> Nd4j.create(data.slice(length / 2, length).toArray), "ds" -> Nd4j.create(ds.slice(length / 2, length).toArray))
    val model = Prophet()
    model.fit(history)
    model.predict(future)
  }

  test("Prophet.fitPredictNoSeason") {
    val history: Map[String, INDArray] = Map("y" -> Nd4j.create(data.slice(0, length / 2).toArray), "ds" -> Nd4j.create(ds.slice(0, length / 2).toArray))
    val future: Map[String, INDArray] = Map("y" -> Nd4j.create(data.slice(length / 2, length).toArray), "ds" -> Nd4j.create(ds.slice(length / 2, length).toArray))
    val model = Prophet(dailySeasonality = false, weeklySeasonality = false, yearlySeasonality = false)
    model.fit(history)
    model.predict(future)
  }

  test("Prophet.fitPredictNoChangePoints") {
    val history: Map[String, INDArray] = Map("y" -> Nd4j.create(data.slice(0, length / 2).toArray), "ds" -> Nd4j.create(ds.slice(0, length / 2).toArray))
    val future: Map[String, INDArray] = Map("y" -> Nd4j.create(data.slice(length / 2, length).toArray), "ds" -> Nd4j.create(ds.slice(length / 2, length).toArray))
    val model = Prophet(nChangePoints = 0)
    model.fit(history)
    model.predict(future)
  }
}
