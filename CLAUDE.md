# facekit — Claude Code 작업 규율

## 성격
공개 자료(공개 모델·논문·공식 문서)만으로 작성하는 클린룸 구현.

## 절대 규칙
1. 외부 독점/사내 코드를 참조·복사하지 않는다.
2. core/ 에 package:flutter/*, dart:ui 를 import 하지 않는다.
3. 수학·판정 로직은 부수효과 없는 순수 함수.
4. 모델 호출은 contracts.dart 인터페이스 뒤에 둔다.
5. 무거운 추론은 isolate에서 실행한다.
6. 임계값 등 수치는 새로 튜닝한다(옛 값 반입 금지).

## 라이선스
- 검출(BlazeFace Apache2.0 / YuNet MIT): 동봉 가능.
- 임베딩: 동봉 금지(BYOM). Demo 모델은 license.redistributable=false 로 표기,
  release 빌드 시 차단.

## 구조 / 계층 의존 방향 (단방향 — 역류 금지)
UI/example → pipeline → detection/alignment/embedding/matching → inference → core

core는 아무것도 의존하지 않는다 (Flutter조차).

## 검증
core/ 에서 `grep "package:flutter"` 결과가 0줄이어야 한다.
