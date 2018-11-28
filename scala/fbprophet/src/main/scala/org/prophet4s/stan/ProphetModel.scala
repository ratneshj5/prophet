package org.prophet4s.stan

import java.io.InputStream

import com.cibo.scalastan._

import scala.io.Codec

/**
  * User: ratnesh
  * Date: 14/11/18
  * Time: 5:02 PM
  */

class ProphetModel extends StanModel {

  val T: DataDeclaration[StanInt] = data(int()) // Number of time periods
  val K: DataDeclaration[StanInt] = data(int(lower = 1)) // Number of regressors
  val t: DataDeclaration[StanVector] = data(vector(T)) // Time
  val cap: DataDeclaration[StanVector] = data(vector(T)) // Capacities for logistic trend
  val y: DataDeclaration[StanVector] = data(vector(T)) // Time series
  val S: DataDeclaration[StanInt] = data(int()) // Number of changepoints
  val t_change: DataDeclaration[StanVector] = data(vector(S)) // Times of trend changepoints
  val X: DataDeclaration[StanMatrix] = data(matrix(T, K)); // Regressors
  val sigmas: DataDeclaration[StanVector] = data(vector(K)) // Scale on seasonality prior
  val tau: DataDeclaration[StanReal] = data(real(lower = 0.0)); // Scale on changepoints prior
  val trend_indicator: DataDeclaration[StanInt] = data(int()) // 0 for linear, 1 for logistic
  val s_a: DataDeclaration[StanVector] = data(vector(K)) // Indicator of additive features
  val s_m: DataDeclaration[StanVector] = data(vector(K)) // Indicator of multiplicative features


  val k: ParameterDeclaration[StanReal] = parameter(real()) // Base trend growth rate
  val m: ParameterDeclaration[StanReal] = parameter(real()) // Trend offset
  val delta: ParameterDeclaration[StanVector] = parameter(vector(S)) // Trend rate adjustments
  val sigma_obs: ParameterDeclaration[StanReal] = parameter(real(lower = 0.0)); // Observation noise
  val beta: ParameterDeclaration[StanVector] = parameter(vector(K)) // Regressor coefficients

  val stream: InputStream = getClass.getResourceAsStream("/stan/unix/prophet.stan")
  val code: String = scala.io.Source.fromInputStream(stream)(Codec.UTF8).getLines.mkString("\n")
  loadFromString(code)
}


