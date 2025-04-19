import SwiftUI
import WidgetKit

@main
struct BankWidgetBundle: WidgetBundle {
    // Initialize the AccountsStore when the app launches
    init() {
        // Ensure the AccountsStore is initialized
        _ = AccountsStore.shared
    }
    
    var body: some Widget {
        AccountPreviewWidget()
        PaymentWidget()
    }
} 
