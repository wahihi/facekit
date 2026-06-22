// Orchestrates the full face recognition pipeline.
// Heavy inference runs in a Dart isolate to avoid dropping UI frames.
// Source: design spec §7 Pipeline + §5 R5.

import 'dart:isolate';
import '../core/contracts.dart';
import '../core/models.dart';

/// Full pipeline: detect → align → embed → match.
///
/// Construct once and reuse. Call [dispose] when done.
class FacePipeline {
  final FaceDetector detector;
  final FaceAligner aligner;
  final FaceEmbedder embedder;
  final FaceMatcher matcher;

  const FacePipeline({
    required this.detector,
    required this.aligner,
    required this.embedder,
    required this.matcher,
  });

  /// Identifies the most prominent face in [image] against [gallery].
  ///
  /// Returns null if no face is detected.
  /// Heavy work (embedding inference) runs in an isolate.
  Future<MatchResult?> identify(
    FaceImage image,
    List<Enrollment> gallery,
  ) async {
    final faces = await detector.detect(image);
    if (faces.isEmpty) return null;

    // Use the highest-confidence face only (first after descending sort by score).
    final best = faces.reduce((a, b) => a.score >= b.score ? a : b);
    final aligned = aligner.align(image, best);
    final embedding = await _embedInIsolate(aligned);
    return matcher.match(embedding, gallery);
  }

  /// Enrolls the most prominent face in [image] and returns its embedding.
  ///
  /// Returns null if no face is detected.
  Future<Embedding?> enroll(FaceImage image) async {
    final faces = await detector.detect(image);
    if (faces.isEmpty) return null;

    final best = faces.reduce((a, b) => a.score >= b.score ? a : b);
    final aligned = aligner.align(image, best);
    return _embedInIsolate(aligned);
  }

  /// Runs [embedder.embed] in a separate isolate.
  ///
  /// Isolate communication uses [_IsolateEmbedRequest] to pass the aligned
  /// face bytes and a [SendPort] for the result. The embedder interface is
  /// not directly serialisable, so we run inference inside a closure that
  /// captures [embedder] (valid because the isolate shares heap in Dart 2+
  /// when using [Isolate.run]).
  Future<Embedding> _embedInIsolate(AlignedFace face) async {
    // Isolate.run is the idiomatic way since Dart 2.19 — spawns, runs, returns.
    return Isolate.run(() => embedder.embed(face));
  }
}

/// Lightweight pipeline that runs everything on the calling isolate.
/// Use in tests or when the caller already manages isolate scheduling.
class SyncFacePipeline {
  final FaceDetector detector;
  final FaceAligner aligner;
  final FaceEmbedder embedder;
  final FaceMatcher matcher;

  const SyncFacePipeline({
    required this.detector,
    required this.aligner,
    required this.embedder,
    required this.matcher,
  });

  Future<MatchResult?> identify(FaceImage image, List<Enrollment> gallery) async {
    final faces = await detector.detect(image);
    if (faces.isEmpty) return null;

    final best = faces.reduce((a, b) => a.score >= b.score ? a : b);
    final aligned = aligner.align(image, best);
    final embedding = await embedder.embed(aligned);
    return matcher.match(embedding, gallery);
  }

  Future<Embedding?> enroll(FaceImage image) async {
    final faces = await detector.detect(image);
    if (faces.isEmpty) return null;

    final best = faces.reduce((a, b) => a.score >= b.score ? a : b);
    final aligned = aligner.align(image, best);
    return embedder.embed(aligned);
  }
}
