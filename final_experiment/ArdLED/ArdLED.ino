String myCMD;


void setup() {
  // put your setup code here, to run once:
Serial.begin(115200);
}

void loop() {
  // put your main code here, to run repeatedly:
while(Serial.available()==0){
pinMode(13,OUTPUT);

}
myCMD = Serial.readStringUntil('\r');
Serial.println(myCMD);
if (myCMD == "ON"){
  digitalWrite(13,HIGH);
}
if (myCMD == "OFF"){
  digitalWrite(13,LOW);
}
}
