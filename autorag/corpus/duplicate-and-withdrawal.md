# Duplicate Charges and Withdrawing a Dispute

Two common low-complexity cases — duplicate charges and cardholder withdrawals — have
specific handling that prevents wasted chargebacks.

- A **duplicate charge** (the same merchant, amount, and date posted twice) is disputed
  under reason code **12.6.1 - Duplicate Processing**. Before filing, confirm the two
  entries are not a legitimate authorization plus settlement or two genuinely separate
  purchases.
- If the merchant has already refunded one of the duplicates, the dispute is **closed as
  resolved**; filing a chargeback on an already-refunded charge can cause a double credit
  that is later clawed back.
- A cardholder may **withdraw** a dispute at any time before it is resolved. On
  withdrawal, any provisional credit is reversed and the cardholder is notified. A
  withdrawn dispute can be refiled within the original filing window if circumstances
  change.
- If the cardholder recognizes the charge after filing (common with unfamiliar merchant
  descriptors), treat it as a withdrawal, not a loss, and note the merchant's
  **descriptor** so the same false dispute is not raised again.

Resolving duplicates and withdrawals quickly keeps chargeback ratios low, which protects
merchant relationships and network standing.
