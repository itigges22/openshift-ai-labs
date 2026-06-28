# Chargeback Reason Codes

A reason code classifies why a transaction is being disputed. Using the correct code
is required for the chargeback to be accepted by the card network.

- **10.4 — Other Fraud, Card Absent Environment:** the cardholder did not authorize a
  card-not-present (online or phone) transaction. Most e-commerce fraud uses 10.4.
- **13.1 — Merchandise / Services Not Received:** the cardholder paid but never got
  the goods or service.
- **13.3 — Not as Described or Defective Merchandise:** the item arrived but is
  materially different from its description, or is broken.
- **13.6 — Credit Not Processed:** the merchant agreed to a refund but never issued it.
- **12.5 — Incorrect Amount:** the cardholder was charged a different amount than
  authorized.

Reason code **10.4** requires that no chip-and-PIN authentication was present. If 3-D
Secure (3DS) authentication succeeded, liability usually shifts to the issuer and a
10.4 chargeback will be rejected. Always check the authorization log for a 3DS flag
before filing 10.4.
