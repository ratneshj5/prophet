Prophet: Automatic Forecasting Procedure
========================================

Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

Prophet is `open source software <https://code.facebook.com/projects/>`_ released by Facebook's `Core Data Science team <https://research.fb.com/category/data-science/>`_.

Full documentation and examples available at the homepage: https://facebook.github.io/prophet/

Important links
---------------

- HTML documentation: https://facebook.github.io/prophet/docs/quick_start.html
- Issue tracker: https://github.com/facebook/prophet/issues
- Source code repository: https://github.com/facebook/prophet
- Implementation of Prophet in R: https://cran.r-project.org/package=prophet


Other forecasting packages
--------------------------

- Rob Hyndman's `forecast package <http://robjhyndman.com/software/forecast/>`_
- `Statsmodels <http://statsmodels.sourceforge.net/>`_


Installation
------------

::
Add following in your build.sbt
```sbtshell
libraryDependencies += "com.sprinklr.intuition" %% "fbprophet_2.12" % "0.3"
```

Note:  Installation requires CMDStan, which has its `own installation instructions <https://github.com/stan-dev/cmdstan/wiki/Getting-Started-with-CmdStan>`_.      

Example usage
-------------
::
```scala
import org.prophet4s.model.Prophet
val m = Prophet()
m.fit(df)  // df is map of INDArray with 'y' and 'ds' as keys
m.predict(future)
```


