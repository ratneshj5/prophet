import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.prophet4s.model.Prophet

/**
  * User: ratnesh
  * Date: 20/11/18
  * Time: 2:03 PM
  */
object ProphetExample extends App {
  val source = com.cibo.scalastan.data.CsvDataSource.fromFile("/Users/ratnesh/Downloads/mcd_clips.csv")
  val model = Prophet()
  val data: Seq[Double] = source.read(model.y, "y")
  val ds: Seq[Double] = source.read(model.t, "ds")
  private val cap: Stream[Double] = Stream.fill(ds.length)(40000d)
  val history: Map[String, INDArray] = Map("y" -> Nd4j.create(data.toArray), "ds" -> Nd4j.create(ds.toArray), "cap" -> Nd4j.create(cap.toArray))
  private val time: Long = System.currentTimeMillis()
  model.fit(history)
  private val time2: Long = System.currentTimeMillis()
  println(time2 - time)
  model.predict(history)

}

