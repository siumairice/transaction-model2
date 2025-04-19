import SwiftUI

// Payment row view
struct PaymentRow: View {
    let payment: Payment
    let isSmallWidget: Bool
    
    var body: some View {
        HStack(spacing: isSmallWidget ? 8 : 12) {
            // Date column with rounded square background
            VStack(alignment: .center, spacing: 0) {
                Text(payment.date.month)
                    .font(.system(size: isSmallWidget ? 10 : 12, weight: .medium))
                    .foregroundColor(.secondary)
                Text(payment.date.day)
                    .font(.system(size: isSmallWidget ? 14 : 16, weight: .semibold))
                    .foregroundColor(.secondary)
            }
            .frame(width: isSmallWidget ? 35 : 45, height: isSmallWidget ? 35 : 45)
            .background(Color(.systemGray6))
            .cornerRadius(isSmallWidget ? 8 : 10)
            
            // Main content and amount
            if isSmallWidget {
                // Simplified layout for small widget
                VStack(alignment: .leading, spacing: 2) {
                    Text(payment.merchant)
                        .font(.system(size: 14, weight: .semibold))
                    
                    HStack {
                        Text(payment.description)
                            .font(.system(size: 12))
                            .foregroundColor(.secondary)
                        
                        Spacer()
                        
                        Text(payment.amount)
                            .font(.system(size: 14, weight: .semibold))
                    }
                }
            } else {
                // Standard layout for medium and large widgets
                VStack(alignment: .leading, spacing: 4) {
                    Text(payment.merchant)
                        .font(.system(size: 17, weight: .semibold))
                    Text(payment.description)
                        .font(.system(size: 15))
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text(payment.amount)
                        .font(.system(size: 17, weight: .semibold))
                    Text(payment.paymentMethod)
                        .font(.system(size: 13))
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding(.vertical, isSmallWidget ? 8 : 12)
        .padding(.horizontal, isSmallWidget ? 8 : 12)
    }
} 