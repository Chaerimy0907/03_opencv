void setup() {

  pinMode(5, OUTPUT);              // 5번핀을 출력모드로 설정합니다.

  pinMode(6, OUTPUT);              // 6번핀을 출력모드로 설정합니다.

}

void loop() {

  analogWrite(5, 0);                   // 5번핀에 0(0V)의 신호를 출력합니다.

  analogWrite(6, 150);              // 6번핀에 150(약 3V)의 신호를 출력합니다.

  delay(3000);                           // 3초간 대기

  analogWrite(5, 150);             // 5번핀에 150(약 3V)의 신호를 출력합니다.

  analogWrite(6, 0);                 // 6번핀에 0(0V)의 신호를 출력합니다.

  delay(3000);                         // 3초간 대기

}