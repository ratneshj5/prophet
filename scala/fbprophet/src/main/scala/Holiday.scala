/**
  * User: ratnesh
  * Date: 19/11/18
  * Time: 1:13 PM
  */
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
