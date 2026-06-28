# ATM Withdrawal Disputes

Disputes about ATM transactions follow Regulation E but have their own evidence and
timing nuances because of the physical-terminal record.

- An ATM dispute (cash not dispensed, partial dispense, or wrong amount) must be
  reported within **60 days** of the statement, like other Reg E disputes.
- The first step is to pull the terminal's **journal roll / electronic audit log**,
  which records whether the cash cassette dispensed and whether the bill counter
  detected a jam. This log is the deciding evidence.
- For a **cash-not-dispensed** claim, if the terminal balanced at end of day (no cash
  overage), the dispute is usually denied; an overage equal to the disputed amount
  supports the cardholder.
- Surcharge-only disputes (the cardholder objects to the ATM operator fee) are **not**
  chargeback-eligible and must be taken up with the ATM operator directly.
- Provisional credit timing for ATM disputes is the standard **10 business days**.

ATM disputes at a foreign or out-of-network terminal can take up to **90 days** to
investigate because the terminal log must be requested from the acquiring bank.
