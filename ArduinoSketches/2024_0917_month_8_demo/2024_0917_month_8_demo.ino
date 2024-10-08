#include <thermistor.h>

thermistor therm1(A2,0);  // Analog Pin which is connected to the 3950 temperature sensor, and 0 represents TEMP_SENSOR_0 (see configuration.h for more information).

// Pin definition
const int readerPin = A0; // Change this to any pin you want to use
int analogValue = 0; 
int tempValue = 0;
// Pulse duration definitions (in milliseconds)
const int measurementsamplingrate = 500; // .5 seconds 
const int reportrate = 1000;  // 1 second seconds
const int id = 2;



void setup() {
  Serial.begin(115200);
  randomSeed(2);

}

void loop() {
  // Read the analog value from pin A0 (0 to 1023)
  analogValue = analogRead(readerPin);
  double temp = therm1.analog2temp(); // read temperature
  //Print temperature in port serial

  // Print the value to the Serial Monitor
  Serial.print("2");
  Serial.print(",");
  Serial.print(analogValue);
  // Serial.print(sudoStrain);
  Serial.print(",");
  Serial.println((String)temp);
  // Serial.println(tempValue);
  // Wait for 1 second (1000 milliseconds)
  delay(1000);
}
