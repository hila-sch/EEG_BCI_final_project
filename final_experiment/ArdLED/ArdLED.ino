String myCmd;


void setup() {
  // put your setup code here, to run once:
Serial.begin(115200);
}

void loop() {
  // put your main code here, to run repeatedly:
while(Serial.available()==0){
pinMode(13,OUTPUT);

}
myCmd = Serial.readStringUntil('\r');
Serial.println(myCmd);
if (myCMD == "ON"){
  digitalWrite(13,HIGH);
}
if (myCMD == "OFF"){
  digitalWrite(13,LOW);
}
}