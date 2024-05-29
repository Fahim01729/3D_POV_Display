#include <ESP32Servo.h>
#include <Arduino.h>

// Define pin connections
static const int encoderPinA = 2; // Pin for encoder input A
static const int encoderPinB = 3; // Pin for encoder input B
static const int buttonPin = 6; // Button for starting/stopping the motor
static const int servoPin = 11; // Pin for the servo control

// Servo control ranges
static const int minPulse = 544;
static const int maxPulse = 2400;
static const int neutralPulse = 1500;

// Servo and encoder handling
Servo controlServo;
volatile int servoPosition = 1000; // Default position indicating 'off'
volatile bool flagA = false; 
volatile bool flagB = false; 
volatile int encoderTicks = 0; // Track encoder ticks
volatile byte encoderReading = 0;

// LED and control states
const int ledPin = 13;
bool motorActive = false; // Tracks whether the motor is active
int initialSpeed = 1212; // Initial speed for motor start
int loopCounter = 0;

// Variables for time measurement
const int encoderTicksPerRotation = 360; // Example value, change according to your encoder
volatile int ticksSinceLastRotation = 0;
unsigned long startTime = 0;
unsigned long endTime = 0;
volatile bool rotationComplete = false;

void setup() {
  // Setup encoder inputs
  pinMode(encoderPinA, INPUT_PULLUP);
  pinMode(encoderPinB, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(encoderPinA), readEncoderA, RISING);
  attachInterrupt(digitalPinToInterrupt(encoderPinB), readEncoderB, RISING);

  // Setup button and LED
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(ledPin, OUTPUT);

  // Initialize servo
  controlServo.attach(servoPin);
  controlServo.writeMicroseconds(neutralPulse);
  Serial.begin(19200);
}

void loop() {
  loopCounter++;
  if (loopCounter >= 20000) {
    Serial.println(encoderTicks);
    loopCounter = 0;
  }

  // Maintain encoder position within the servo's operational range
  encoderTicks = constrain(encoderTicks, minPulse, maxPulse);
  controlServo.writeMicroseconds(encoderTicks);

  // Check button state for starting or stopping the motor
  if (digitalRead(buttonPin) == LOW) {
    if (!motorActive) {
      encoderTicks = initialSpeed; // Engage motor at a set speed
      motorActive = true;
      digitalWrite(ledPin, HIGH);
      delay(400);
    } else {
      encoderTicks = neutralPulse; // Disengage motor
      motorActive = false;
      digitalWrite(ledPin, LOW);
      delay(400);
    }
  }

  // Check if a rotation is complete
  if (rotationComplete) {
    endTime = millis();
    unsigned long rotationTime = endTime - startTime;
    Serial.print("Time per rotation: ");
    Serial.print(rotationTime);
    Serial.println(" ms");

    // Reset for the next rotation
    startTime = millis();
    ticksSinceLastRotation = 0;
    rotationComplete = false;
  }
}

// Interrupt handler for encoder pin A
void readEncoderA() {
  noInterrupts(); // Disable interrupts for stable pin read
  encoderReading = LEND & 0xC;
  if (encoderReading == B00001100 && flagA) {
    encoderTicks--;
    flagB = false;
    flagA = false;
    handleEncoderTick();
  } else if (encoderReading == B00000100) {
    flagB = true;
  }
  interrupts(); // Re-enable interrupts
}

// Interrupt handler for encoder pin B
void readEncoderB() {
  noInterrupts();
  encoderReading = LEND & 0xC;
  if (encoderReading == B00001100 && flagB) {
    encoderTicks++;
    flagB = false;
    flagA = false;
    handleEncoderTick();
  } else if (encoderReading == B00001000) {
    flagA = true;
  }
  interrupts();
}

// Handle each encoder tick
void handleEncoderTick() {
  ticksSinceLastRotation++;
  if (ticksSinceLastRotation >= encoderTicksPerRotation) {
    rotationComplete = true;
  }
}
