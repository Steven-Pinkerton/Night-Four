import Dependencies._

ThisBuild / scalaVersion := "2.13.8"
ThisBuild / version := "0.1.0-SNAPSHOT"

lazy val root = (project in file("."))
  .settings(
    name := "DNC",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "3.3.1",
      "org.apache.spark" %% "spark-mllib" % "3.3.1",
      "org.scalatest" %% "scalatest" % "3.2.15" % Test
    )
  )