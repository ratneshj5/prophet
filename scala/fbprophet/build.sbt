name := "fbprophet"

version := "0.3"

scalaVersion := "2.12.7"

organization := "com.sprinklr.intuition"

resolvers += Resolver.bintrayRepo("cibotech", "public")
resolvers += "Nexus" at "https://host/nexus/content/repositories/thirdparty/"

libraryDependencies += "com.cibo" %% "scalastan" % "0.8.1"
// https://mvnrepository.com/artifact/org.nd4j/nd4j-native-platform
libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.9.1"
// https://mvnrepository.com/artifact/org.nd4j/nd4j-api
libraryDependencies += "org.nd4j" % "nd4j-scala-api" % "0.0.3.5.5.5"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.5" % "test"

libraryDependencies += "org.apache.commons" % "commons-math3" % "3.6.1"
fork := true
envVars := Map("CMDSTAN_HOME" -> "~/cmdstan-2.18.0")
testOptions in Test += Tests.Argument("-oD")

publishMavenStyle := true
credentials += Credentials(realm = "Sonatype Nexus Repository Manager", host = "host", userName = "userName", passwd = "passwd")
publishTo := Some("Sonatype Snapshots Nexus" at "https://host/nexus/content/repositories/thirdparty/")
