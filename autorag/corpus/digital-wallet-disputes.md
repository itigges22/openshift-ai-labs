# Digital Wallet and Tokenized Payment Disputes

Transactions made through Apple Pay, Google Pay, or a merchant-stored token add a layer
of authentication that changes how disputes are handled.

- A digital-wallet transaction is authenticated with a **device token and biometric or
  passcode**, which usually counts as strong customer authentication. This makes a pure
  "I didn't authorize it" (10.4) claim hard to win, because liability often shifts to
  the cardholder or device holder.
- If a device was **lost or stolen and the wallet was used before the card was frozen**,
  treat it as card-present fraud: the cardholder must report within **2 business days**
  to keep liability capped at **$50** under Reg E.
- Account-takeover cases (a fraudster provisioned the card into *their own* wallet) are
  treated as true fraud and are **not** subject to the device-holder liability shift;
  these require the account-security team's confirmation of the unauthorized provisioning.
- Tokenized recurring charges still follow the normal cancellation rules: cancel with
  the merchant first, then dispute charges billed after cancellation.

Always check whether the token was provisioned by the genuine cardholder before
classifying a digital-wallet dispute.
