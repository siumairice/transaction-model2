import SwiftUI

// MARK: - Reusable UI Components
struct ToggleVisibilityButton: View {
    let isHidden: Bool
    
    var body: some View {
        if #available(iOSApplicationExtension 17.0, *) {
            Button(intent: ToggleBalanceVisibilityIntent()) {
                Image(systemName: isHidden ? "eye.slash" : "eye")
                    .font(.system(size: 16))
                    .foregroundColor(.gray)
            }
        }
    }
}

struct RefreshButton: View {
    var body: some View {
        if #available(iOSApplicationExtension 17.0, *) {
            Button(intent: RefreshWidgetIntent()) {
                Image(systemName: "arrow.clockwise")
                    .font(.system(size: 16))
                    .foregroundColor(.gray)
            }
        }
    }
} 