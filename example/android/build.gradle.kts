allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")

    // tflite_flutter's own android/build.gradle hardcodes Java 11 source/target
    // compatibility but doesn't pin a matching Kotlin jvmTarget, so it defaults
    // to this toolchain's JDK (17) — tripping AGP's "inconsistent JVM target"
    // check. Force every subproject (including plugins) to compile at a
    // consistent Java/Kotlin target instead of patching the plugin itself.
    // :app already sets Java/Kotlin 17 itself in its own build.gradle.kts, and
    // evaluationDependsOn(":app") above means :app is already evaluated by
    // the time we get here — calling afterEvaluate on it would throw. Only
    // patch the *other* subprojects (plugins like tflite_flutter).
    if (project.name != "app") {
        afterEvaluate {
            extensions.findByType<com.android.build.gradle.BaseExtension>()?.apply {
                compileOptions {
                    sourceCompatibility = JavaVersion.VERSION_17
                    targetCompatibility = JavaVersion.VERSION_17
                }
            }
            tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
                compilerOptions {
                    jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
                }
            }
        }
    }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
