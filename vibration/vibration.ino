String myCMD;

void setup() {
  // put your setup code here, to run once:
Serial.begin(115200);
}

void loop() {
  // put your main code here, to run repeatedly:
while(Serial.available()==0){
pinMode(7,OUTPUT);

}
myCMD = Serial.readStringUntil('\r');
Serial.println(myCMD);
if (myCMD == "ON"){
  digitalWrite(7,HIGH);
}
if (myCMD == "OFF"){
  digitalWrite(7,LOW);
}
}
