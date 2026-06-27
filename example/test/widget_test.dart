// Basic smoke test: the app builds without throwing and shows its title.
//
// Pre-existing note: this file was unmodified `flutter create` boilerplate
// referencing a non-existent `MyApp` counter widget (this app is
// `FacekitExampleApp`/`RecognitionPage`) — it never compiled. Fixed in
// passing while touching this app for the liveness/overlay work.
import 'package:flutter_test/flutter_test.dart';

import 'package:facekit_example/main.dart';

void main() {
  testWidgets('app builds and shows its title', (WidgetTester tester) async {
    await tester.pumpWidget(const FacekitExampleApp());
    await tester.pump();

    expect(find.text('facekit example'), findsOneWidget);
  });
}
