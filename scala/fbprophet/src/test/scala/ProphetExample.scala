

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
  val history: Map[String, Stream[Any]] = Map("y" -> data.toStream, "ds" -> ds.toStream)
  model.fit(history)



}

