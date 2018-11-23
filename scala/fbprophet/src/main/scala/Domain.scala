/**
  * User: ratnesh
  * Date: 19/11/18
  * Time: 1:13 PM
  */
object Domain {

  /*
case class with columns holiday (string) and ds (date type)
        and optionally columns lower_window and upper_window which specify a
        range of days around the date to be included as holidays.
        lower_window=-2 will include 2 days prior to the date as holidays. Also
        optionally can have a column prior_scale specifying the prior scale for
        that holiday.
 */
  case class Holiday(
                      holiday: String,
                      ds: Long,
                      lower_window: Option[Int],
                      upper_window: Option[Int],
                      prior_scale: Option[Double]
                    )

  case class Seasonality(name: String, period: Double, fourierOrder: Int, priorScale: Double, mode: String = "additive")

  case class Regressor(name: String, var mu: Double = 0, var std: Double = 1, standardize: Option[Boolean], priorScale: Double, mode: String = "additive")

  case class SeasonalData(seasonalFeatures: Stream[Stream[Double]], prior_scales: Stream[Double], s_a: Stream[Double], s_m: Stream[Double])

  case class Parameter(k: Double, m: Double, delta: Seq[Double], sigma_obs: Double, beta: Seq[Double]) {
    override def toString: String = s"{k : $k, m : $m, delta : $delta, sigma_obs : $sigma_obs, beta : $beta}"
  }

}