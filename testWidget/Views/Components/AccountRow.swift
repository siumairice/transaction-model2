import SwiftUI

struct AccountRowView: View {
    let viewModel: AccountViewModel
    
    init(account: AccountDetail, isHidden: Bool) {
        self.viewModel = AccountViewModel(account: account, isHidden: isHidden)
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(viewModel.account.accountName)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(.black)
                
                Spacer()
                
                Text(viewModel.displayBalance)
                    .font(.system(size: 14, weight: .bold))
                    .foregroundColor(.black)
                
                Text(viewModel.currency)
                    .font(.system(size: 10))
                    .foregroundColor(.gray)
                    .padding(.leading, -4)
            }
            
            HStack {
                Text("\(viewModel.accountType) (\(viewModel.account.accountNumber))")
                    .font(.system(size: 12))
                    .foregroundColor(.gray)
                
                Spacer()
                
                Text(viewModel.displayMonthlyChange)
                    .font(.system(size: 12))
                    .foregroundColor(viewModel.monthlyChangeColor)
                
                Text("in \(viewModel.month)")
                    .font(.system(size: 12))
                    .foregroundColor(.gray)
            }
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 12)
    }
} 