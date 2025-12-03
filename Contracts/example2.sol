// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// NON-VULNERABLE (but DEPRECATED): Using .transfer()
contract OldSecureEtherStore {
    mapping(address => uint) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint _amount) public {
        require(balances[msg.sender] >= _amount, "Insufficient balance");

        balances[msg.sender] -= _amount;

        // THE (OLD) FIX: .transfer() only forwards 2300 gas.
        // This is not enough gas for an attacker to re-enter.
        // WARNING: Do not use this method in modern contracts.
        payable(msg.sender).transfer(_amount);
    }
}