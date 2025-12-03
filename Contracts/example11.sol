// SPDX-License-Identifier: MIT
pragma solidity 0.8.0;

/**
 * @title SecureTimelockVault
 * @dev A vault that locks Ether for a set time.
 * The withdraw function is reentrancy-safe.
 */
contract SecureTimelockVault {

    struct Lock {
        uint amount;
        uint unlockTime;
    }
    
    mapping(address => Lock) public locks;

    event Locked(address indexed user, uint amount, uint unlockTime);
    event Withdrawn(address indexed user, uint amount);

    /**
     * @dev Deposit and lock funds for a duration.
     */
    function deposit(uint _lockDurationInSeconds) public payable {
        require(msg.value > 0, "Deposit must be positive");
        require(locks[msg.sender].amount == 0, "You already have a lock");
        
        locks[msg.sender] = Lock({
            amount: msg.value,
            unlockTime: block.timestamp + _lockDurationInSeconds
        });
        
        emit Locked(msg.sender, msg.value, locks[msg.sender].unlockTime);
    }

    /**
     * @dev Securely withdraw funds *after* the timelock expires.
     */
    function withdraw() public {
        
        // 1. CHECKS
        Lock storage userLock = locks[msg.sender];
        uint amount = userLock.amount;
        
        require(amount > 0, "No funds locked");
        require(block.timestamp >= userLock.unlockTime, "Lock period not over");

        // 2. EFFECTS (THE FIX)
        // The lock is deleted *before* the Ether is sent.
        userLock.amount = 0;
        userLock.unlockTime = 0;
        // A more robust way: delete locks[msg.sender];

        emit Withdrawn(msg.sender, amount);

        // 3. INTERACTION
        (bool sent, ) = msg.sender.call{value: amount}("");
        require(sent, "Withdrawal failed");
    }
}