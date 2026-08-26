// Pure-Dart stand-ins for the handful of package:flutter/foundation.dart
// symbols this codebase used purely for their values, not for anything
// Flutter-engine-specific — same definitions foundation.dart itself uses
// internally. Importing this instead keeps layers outside core/ importable
// from a plain Dart context (see doc/KR/linux_port_scope.md, S1).
export 'package:meta/meta.dart' show visibleForTesting;

const bool kReleaseMode = bool.fromEnvironment('dart.vm.product');
const bool kProfileMode = bool.fromEnvironment('dart.vm.profile');
const bool kDebugMode = !kReleaseMode && !kProfileMode;

/// Prints [message], mirroring package:flutter/foundation.dart's debugPrint
/// (minus its release-mode truncation/throttling, neither of which this
/// codebase's call sites rely on).
void debugPrint(String? message) {
  // ignore: avoid_print
  if (message != null) print(message);
}
