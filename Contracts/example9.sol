// SPDX-License-Identifier: MIT
// WARNING: VULNERABLE CODE!
pragma solidity ^0.8.0;

/**
 * @title VulnerableRental
 * @dev Manages security deposits for rentals.
 *
 * THE FLAW: `returnItemAndClaimDeposit` sends the deposit (Interaction)
 * *before* clearing the renter's status (Effect).
 */
contract VulnerableRental {
    
    uint public constant DEPOSIT_AMOUNT = 1 ether;
    address public owner;
    
    mapping(address => bool) public isRenting;
    
    event Rented(address indexed renter);
    event Returned(address indexed renter);

    constructor() {
        owner = msg.sender;
    }

    /**
     * @dev Pay deposit to rent the item.
     */
    function rent() public payable {
        require(!isRenting[msg.sender], "Already renting");
        require(msg.value == DEPOSIT_AMOUNT, "Incorrect deposit");
        
        isRenting[msg.sender] = true;
        emit Rented(msg.sender);
    }

    /**
     * @dev The VULNERABLE function to return item and get deposit back.
     */
    function returnItemAndClaimDeposit() public {
        
        // 1. CHECKS
        require(isRenting[msg.sender], "You are not renting");

        // 2. INTERACTION (THE FLAW)
        // The deposit is sent *before* 'isRenting' is set to false.
        // Attacker's 'receive()' hook can call this function
        // again, and the check on line 51 will pass.
        (bool sent, ) = msg.sender.call{value: DEPOSIT_AMOUNT}("");
        require(sent, "Deposit refund failed");

        // 3. EFFECT (TOO LATE)
        // The renter status is only cleared *after* the attacker
        // has received their deposit multiple times.
        isRenting[msg.sender] = false;

        emit Returned(msg.sender);
    }
}