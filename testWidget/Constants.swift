//
//  Constants.swift
//  testWidgetExtension
//
//  Created by Nicole Go on 2025-03-27.
//

import WidgetKit
import SwiftUI

/// Constants for the Banking Widget
enum BankWidgetConstants {
    /// User Defaults Keys
    static let appGroupSuite = "group.com.yourcompany.bankingapp"
    
    enum Keys {
        /// Key for storing the balance visibility state
        static let balancesHidden = "balancesHidden"
        
        /// Key for storing accounts data
        static let accountsData = "accountsData"
    }
}
