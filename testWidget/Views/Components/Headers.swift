import SwiftUI

struct BankingWidgetHeader: View {
    let isHidden: Bool
    var padding: EdgeInsets = EdgeInsets(top: 10, leading: 16, bottom: 10, trailing: 16)
    
    var body: some View {
        HStack {
            Image(systemName: "creditcard.fill")
                .foregroundColor(.yellow)
            
            Spacer()
            
            RefreshButton()
            ToggleVisibilityButton(isHidden: isHidden)
        }
        .padding(padding)
        .background(Color.blue.opacity(0.5))
    }
}

struct SmallWidgetHeader: View {
    let isHidden: Bool
    
    var body: some View {
        HStack {
            Image(systemName: "creditcard.fill")
                .font(.system(size: 14))
                .foregroundColor(.yellow)
            Spacer()
            
            RefreshButton()
            ToggleVisibilityButton(isHidden: isHidden)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 10)
        .background(Color.blue.opacity(0.5))
    }
} 