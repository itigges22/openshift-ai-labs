# Card Replacement and Reissue After Fraud

When a card is compromised, reissue logistics affect both fraud containment and the
cardholder's recurring payments.

- On confirmed fraud, the compromised card number is **blocked immediately** and a new
  card with a new number and security code is issued. Standard delivery is **5–7
  business days**; expedited delivery is **1–2 business days** and may carry a fee that
  is **waived for confirmed fraud**.
- A new card means a new **token**: digital-wallet and merchant-stored credentials must
  be re-provisioned. Card-on-file recurring merchants are updated automatically only if
  the issuer participates in a **card-updater service**; otherwise the cardholder must
  update each merchant.
- **Legitimate recurring charges** (utilities, subscriptions) should be flagged so they
  are not declined during the transition; the assistant can list known recurring
  merchants from the transaction history.
- The cardholder's account number and any linked autopay remain the same; only the card
  credentials change.

Reissuing a card does **not** by itself resolve or withdraw any open dispute; the two
processes are tracked separately.
