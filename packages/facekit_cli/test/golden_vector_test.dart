// S3 DoD (doc/KR/linux_port_scope.md) — does this Linux/facekit_cli pipeline
// produce the same embedding Android does for the same input photo? Catches
// platform-specific bugs (rotation handling, channel order, float rounding
// order) that a same-platform-only test can't see.
//
// test/fixtures/me_1_android_golden.json was captured 2026-09-03 from the
// example app's "골든 벡터 덤프 (S3)" debug button (see example/lib/main.dart,
// _dumpGoldenVector) on a real Pixel 7, running assets/golden/me_1.jpg
// (== tool/model_verification/me_1.jpg, copied byte-for-byte) through the
// device's own YuNet+AuraFace pipeline.
//
// Needs libtensorflowlite_c for this platform findable by TfliteFfiRunner
// (FACEKIT_TFLITE_C_LIBRARY env var, or see bindings/bindings.dart) —
// same pre-existing CI gap as facekit_core's own model-loading smoke tests.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:test/test.dart';

import 'package:facekit_cli/facekit_cli.dart';

void main() {
  test('facekit_cli embedding for me_1.jpg matches the real Android device (cosine >= 0.999)', () async {
    final golden = jsonDecode(
      File('test/fixtures/me_1_android_golden.json').readAsStringSync(),
    ) as Map<String, dynamic>;
    final goldenFace = (golden['faces'] as List).first as Map<String, dynamic>;
    final goldenEmbedding = (goldenFace['embedding'] as List).cast<num>().map((n) => n.toDouble()).toList();

    final results = await runDetectAlignEmbed(
      imagePath: '../../tool/model_verification/me_1.jpg',
      detectorDir: '../../assets/models/yunet_160',
      embedderDir: '../../example/assets/models/auraface',
    );

    expect(results, hasLength(1));
    final embedding = results.first.embedding;
    expect(embedding, isNotNull, reason: '정렬 실패 — 골든 비교 불가');

    final cosine = _cosine(embedding!, goldenEmbedding);
    // AuraFace's own genuine-pair threshold is 0.211 (see
    // example/assets/models/auraface/manifest.json) — 0.999 here is a much
    // tighter bar checking the *same* pipeline, not just the same identity.
    expect(cosine, greaterThanOrEqualTo(0.999),
        reason: 'Linux/Android 임베딩이 갈라짐 — 회전/채널 순서/전처리 순서를 의심해볼 것');
  });
}

double _cosine(List<double> a, List<double> b) {
  var dot = 0.0, na = 0.0, nb = 0.0;
  for (var i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (math.sqrt(na) * math.sqrt(nb));
}
