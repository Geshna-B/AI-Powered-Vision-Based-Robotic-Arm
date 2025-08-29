

#include <Servo.h>
#include<SoftwareSerial.h>
Servo thumb;  
Servo ring;  
Servo middle;
Servo point;
Servo little; 

int pos = 0;   
char data = 0;
int Open = 0;
int Close = 120;
int Topen = 0;
int Tclose = 150;
#define numOfValsRec 5
#define digitsPerValRec 1

int valsRec[numOfValsRec];
int stringLength = numOfValsRec * digitsPerValRec + 1;
int counter = 0;
bool counterStart = false;
String receivedString;

void setup()
{
  Serial.begin(9600);
  point.attach(2); point.write(Close);
  middle.attach(4); middle.write(Close);
  ring.attach(3); ring.write(Close);
  little.attach(11);  little.write(Close);
  thumb.attach(6); thumb.write(Close);
  delay(1000);
}

void receiveData()
{
  while (Serial.available())
  {
    char c = Serial.read();
    if (c == '$')
    {
      counterStart = true;
    }
    if (counterStart)
    {
      if (counter < stringLength)
      {
        receivedString = String(receivedString + c);
        counter++;
      }
      if (counter >= stringLength)
      {
        for (int i = 0; i < numOfValsRec; i++)
        {
          int num = (i * digitsPerValRec) + 1;
          valsRec[i] = receivedString.substring(num, num + digitsPerValRec).toInt();
        }
        receivedString = "";
        counter = 0;
        counterStart = false;
      }

    }
  }
}

void loop()
{
  receiveData();

  if (valsRec[0] == 1) {
    thumb.write(Topen);
  } else {
    thumb.write(Tclose);
  }
  if (valsRec[1] == 1) {
    point.write(Open);
  } else {
    point.write(Close);
  }
  if (valsRec[2] == 1) {
    middle.write(Open);
  } else {
    middle.write(Close);
  }
  if (valsRec[3] == 1) {
    ring.write(Open);
  } else {
    ring.write(Close);
  }
  if (valsRec[4] == 1) {
    little.write(Open);
  } else {
    little.write(Close);
  }
}
