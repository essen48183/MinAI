# Stage0_5

4x1 Q8.8.8 weighted analog MAC tile for Raspberry Pi Zero 2W using six MCP4251 digipots, MCP4728 DAC, ADS1115 ADC, 74HC138 SPI chip-select decode, separate analog/digital 3.3V rails, and a compact 4-layer mixed-signal PCB.

## Component Notes

### C1 — Generic Capacitor
**Role:** Decoupling

Local high-frequency bypass capacitors for the digipots, decoder, DAC, ADC, op amps, regulator, and reference buffer

### C2 — Generic Capacitor
**Role:** Decoupling

Local high-frequency bypass capacitors for the digipots, decoder, DAC, ADC, op amps, regulator, and reference buffer

### C3 — Generic Capacitor
**Role:** Decoupling

Local high-frequency bypass capacitors for the digipots, decoder, DAC, ADC, op amps, regulator, and reference buffer

### C4 — Generic Capacitor
**Role:** Decoupling

Local high-frequency bypass capacitors for the digipots, decoder, DAC, ADC, op amps, regulator, and reference buffer

### C5 — Generic Capacitor
**Role:** Decoupling

Local high-frequency bypass capacitors for the digipots, decoder, DAC, ADC, op amps, regulator, and reference buffer

### C6 — Generic Capacitor
**Role:** Decoupling

Local high-frequency bypass capacitors for the digipots, decoder, DAC, ADC, op amps, regulator, and reference buffer

### C7 — Generic Capacitor
**Role:** Decoupling

Local high-frequency bypass capacitors for the digipots, decoder, DAC, ADC, op amps, regulator, and reference buffer

### C8 — Generic Capacitor
**Role:** Decoupling

Local high-frequency bypass capacitors for the digipots, decoder, DAC, ADC, op amps, regulator, and reference buffer

### C9 — Generic Capacitor
**Role:** Decoupling

Local high-frequency bypass capacitors for the digipots, decoder, DAC, ADC, op amps, regulator, and reference buffer

### C10 — Generic Capacitor
**Role:** Decoupling

Local high-frequency bypass capacitors for the digipots, decoder, DAC, ADC, op amps, regulator, and reference buffer

### C11 — Generic Capacitor
**Role:** Decoupling

Local high-frequency bypass capacitors for the digipots, decoder, DAC, ADC, op amps, regulator, and reference buffer

### C12 — Generic Capacitor
**Role:** Decoupling

Local high-frequency bypass capacitors for the digipots, decoder, DAC, ADC, op amps, regulator, and reference buffer

### C13 — Generic Capacitor
**Role:** Decoupling

Local high-frequency bypass capacitors for the digipots, decoder, DAC, ADC, op amps, regulator, and reference buffer

### C14 — Generic Capacitor
**Role:** Decoupling

Local high-frequency bypass capacitors for the digipots, decoder, DAC, ADC, op amps, regulator, and reference buffer

### C15 — Generic Capacitor
**Role:** Bulk

Bulk bypass capacitors on 5V input, regulator output, analog rail, digital rail, DAC/ADC supply, and buffered midrail reference

### C16 — Generic Capacitor
**Role:** Bulk

Bulk bypass capacitors on 5V input, regulator output, analog rail, digital rail, DAC/ADC supply, and buffered midrail reference

### C17 — Generic Capacitor
**Role:** Bulk

Bulk bypass capacitors on 5V input, regulator output, analog rail, digital rail, DAC/ADC supply, and buffered midrail reference

### C18 — Generic Capacitor
**Role:** Bulk

Bulk bypass capacitors on 5V input, regulator output, analog rail, digital rail, DAC/ADC supply, and buffered midrail reference

### C19 — Generic Capacitor
**Role:** Bulk

Bulk bypass capacitors on 5V input, regulator output, analog rail, digital rail, DAC/ADC supply, and buffered midrail reference

### C20 — Generic Capacitor
**Role:** Bulk

Bulk bypass capacitors on 5V input, regulator output, analog rail, digital rail, DAC/ADC supply, and buffered midrail reference

### FB1 — BLM18AG601SN1D
**Role:** Rail Filter

Filters the analog 3.3V branch from the main digital 3.3V regulator output

### J1 — 302-S401
**Role:** Host Interface Header

Carries 5V, multiple grounds, SPI, I2C, and decoder address GPIO to the Raspberry Pi Zero 2W

### R1 — Generic Resistor
**Role:** Bias and Feedback

Used for op-amp feedback/input paths, buffered 1.65V reference divider, shared MISO pull-up, and decoder enable biasing

### R2 — Generic Resistor
**Role:** Bias and Feedback

Used for op-amp feedback/input paths, buffered 1.65V reference divider, shared MISO pull-up, and decoder enable biasing

### R3 — Generic Resistor
**Role:** Bias and Feedback

Used for op-amp feedback/input paths, buffered 1.65V reference divider, shared MISO pull-up, and decoder enable biasing

### R4 — Generic Resistor
**Role:** Bias and Feedback

Used for op-amp feedback/input paths, buffered 1.65V reference divider, shared MISO pull-up, and decoder enable biasing

### R5 — Generic Resistor
**Role:** Bias and Feedback

Used for op-amp feedback/input paths, buffered 1.65V reference divider, shared MISO pull-up, and decoder enable biasing

### R6 — Generic Resistor
**Role:** Bias and Feedback

Used for op-amp feedback/input paths, buffered 1.65V reference divider, shared MISO pull-up, and decoder enable biasing

### R7 — Generic Resistor
**Role:** Bias and Feedback

Used for op-amp feedback/input paths, buffered 1.65V reference divider, shared MISO pull-up, and decoder enable biasing

### R8 — Generic Resistor
**Role:** Bias and Feedback

Used for op-amp feedback/input paths, buffered 1.65V reference divider, shared MISO pull-up, and decoder enable biasing

### R9 — Generic Resistor
**Role:** Bias and Feedback

Used for op-amp feedback/input paths, buffered 1.65V reference divider, shared MISO pull-up, and decoder enable biasing

### R10 — Generic Resistor
**Role:** Bias and Feedback

Used for op-amp feedback/input paths, buffered 1.65V reference divider, shared MISO pull-up, and decoder enable biasing

### R11 — Generic Resistor
**Role:** Bias and Feedback

Used for op-amp feedback/input paths, buffered 1.65V reference divider, shared MISO pull-up, and decoder enable biasing

### R12 — Generic Resistor
**Role:** Slice Summing Input

Input resistors from each digipot wiper into the MSB, MID, and LSB slice summing amplifiers

### R13 — Generic Resistor
**Role:** Slice Summing Input

Input resistors from each digipot wiper into the MSB, MID, and LSB slice summing amplifiers

### R14 — Generic Resistor
**Role:** Slice Summing Input

Input resistors from each digipot wiper into the MSB, MID, and LSB slice summing amplifiers

### R15 — Generic Resistor
**Role:** Slice Summing Input

Input resistors from each digipot wiper into the MSB, MID, and LSB slice summing amplifiers

### R16 — Generic Resistor
**Role:** Slice Summing Input

Input resistors from each digipot wiper into the MSB, MID, and LSB slice summing amplifiers

### R17 — Generic Resistor
**Role:** Slice Summing Input

Input resistors from each digipot wiper into the MSB, MID, and LSB slice summing amplifiers

### R18 — Generic Resistor
**Role:** Slice Summing Input

Input resistors from each digipot wiper into the MSB, MID, and LSB slice summing amplifiers

### R19 — Generic Resistor
**Role:** Slice Summing Input

Input resistors from each digipot wiper into the MSB, MID, and LSB slice summing amplifiers

### R20 — Generic Resistor
**Role:** Slice Summing Input

Input resistors from each digipot wiper into the MSB, MID, and LSB slice summing amplifiers

### R21 — Generic Resistor
**Role:** Slice Summing Input

Input resistors from each digipot wiper into the MSB, MID, and LSB slice summing amplifiers

### R22 — Generic Resistor
**Role:** Slice Summing Input

Input resistors from each digipot wiper into the MSB, MID, and LSB slice summing amplifiers

### R23 — Generic Resistor
**Role:** Slice Summing Input

Input resistors from each digipot wiper into the MSB, MID, and LSB slice summing amplifiers

### R24 — Generic Resistor
**Role:** Binary Scaling

Implements the two cascaded 1/256 scaling paths from VLSB to VLOW and from VLOW to VOUT

### R25 — Generic Resistor
**Role:** Binary Scaling

Implements the two cascaded 1/256 scaling paths from VLSB to VLOW and from VLOW to VOUT

### R26 — Generic Resistor
**Role:** Pull-Up

Shared SDA and SCL pull-ups to the digital 3.3V rail

### R27 — Generic Resistor
**Role:** Pull-Up

Shared SDA and SCL pull-ups to the digital 3.3V rail

### R28 — Generic Resistor
**Role:** Star Ground Tie

Single AGND-to-DGND star tie. Preserve L2 as a continuous ground reference and do not create any additional AGND/DGND ties.

### TP1 — Test Point 1.8mm
**Role:** Test Point

Buffered VCM probe point

### TP2 — Test Point 1.8mm
**Role:** Test Point

Final VOUT probe point

### TP3 — Test Point 1.8mm
**Role:** Test Point

Intermediate VLOW probe point

### TP4 — Test Point 1.8mm
**Role:** Test Point

VLSB probe point

### TP5 — Test Point 1.8mm
**Role:** Test Point

A3V3 rail probe point

### TP6 — Test Point 1.8mm
**Role:** Test Point

D3V3 rail probe point

### TP7 — Test Point 1.8mm
**Role:** Test Point

AGND probe point

### TP8 — Test Point 1.8mm
**Role:** Test Point

DGND probe point

### U1 — MCP4251-103E-SL
**Role:** Programmable Weight Slice

MSB+MID owner for channel 0 and channel 1 slices; preserve this assignment during placement and routing.

### U2 — MCP4251-103E-SL
**Role:** Programmable Weight Slice

MSB+MID owner for channel 1 and channel 2 slices; preserve this assignment during placement and routing.

### U3 — MCP4251-103E-SL
**Role:** Programmable Weight Slice

MSB+MID owner for channel 2 and channel 3 slices; preserve this assignment during placement and routing.

### U4 — MCP4251-103E-SL
**Role:** Programmable Weight Slice

MSB+MID owner for the remaining high-significance slices; preserve this assignment during placement and routing.

### U5 — MCP4251-103E-SL
**Role:** Programmable Weight Slice

LSB owner for the lower-significance slices only; do not repurpose for MSB or MID weighting.

### U6 — MCP4251-103E-SL
**Role:** Programmable Weight Slice

LSB owner for the lower-significance slices only; do not repurpose for MSB or MID weighting.

### U7 — SN74HC138PWR
**Role:** Chip-Select Decoder

Decodes three Raspberry Pi GPIO address bits plus an enable gate into six active-low digipot chip-selects

### U8 — MCP4728A0T-E/UN
**Role:** Input DAC

Generates the four programmable analog input voltages for the MAC tile

### U9 — ADS1115IDGSR
**Role:** Output ADC

Measures VOUT and the buffered midrail reference during bring-up and closed-loop experiments

### U10 — AMS1117-3.3
**Role:** 3.3V Regulator

Generates the local 3.3V domain, then feeds separated digital and analog branches

### U11 — OPA2333AIDGKTG4
**Role:** Precision Summing Amplifier

Implements the buffered midrail reference plus the MSB, MID, LSB, and cascaded output combining amplifier stages

### U12 — OPA2333AIDGKTG4
**Role:** Precision Summing Amplifier

Implements the buffered midrail reference plus the MSB, MID, LSB, and cascaded output combining amplifier stages

### U13 — OPA2333AIDGKTG4
**Role:** Precision Summing Amplifier

Implements the buffered midrail reference plus the MSB, MID, LSB, and cascaded output combining amplifier stages
