import SwiftUI

struct EmptyAccountsView: View {
    var body: some View {
        VStack {
            Spacer()
            Text("No accounts selected")
                .foregroundColor(.gray)
                .frame(maxWidth: .infinity, alignment: .center)
            Spacer()
        }
    }
} 