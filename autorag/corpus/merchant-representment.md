# Merchant Representment and Pre-Arbitration

After a chargeback is filed, the merchant may respond. Understanding this flow prevents
the assistant from giving cardholders false assurance.

1. **Chargeback filed** by the issuer on the cardholder's behalf.
2. **Representment:** the merchant disputes the chargeback by submitting compelling
   evidence (for example, proof of delivery, a signed receipt, or 3DS authentication).
   The merchant has **45 days** to represent.
3. **Pre-arbitration:** if the issuer still disagrees after representment, it files
   pre-arbitration. The merchant then has **30 days** to accept or escalate.
4. **Arbitration:** the card network makes a binding ruling. The losing side pays an
   arbitration fee (typically **$500**) plus the disputed amount.

Compelling evidence that most often defeats a cardholder dispute includes a matching
AVS (address verification) result, a successful 3DS authentication, prior undisputed
transactions with the same merchant, and proof the digital goods were accessed. The
assistant should warn cardholders that filing a dispute they cannot substantiate can
result in the chargeback being reversed at arbitration.
