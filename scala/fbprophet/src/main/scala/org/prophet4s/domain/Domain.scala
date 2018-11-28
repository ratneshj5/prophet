package org.prophet4s.domain

import org.nd4j.linalg.api.ndarray.INDArray

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

  case class SeasonalData(seasonalFeatures: INDArray, prior_scales: Stream[Double], s_a: INDArray, s_m: INDArray)

  case class Parameter(k: INDArray, m: INDArray, delta: INDArray, sigma_obs: INDArray, beta: INDArray) {
    override def toString: String = s"{k : $k, m : $m, delta : $delta, sigma_obs : $sigma_obs, beta : $beta}"
  }

}
