import { Router, type IRouter } from "express";
import * as fs from 'fs';
import * as path from 'path';
import crypto from 'crypto';

const router: IRouter = Router();

// Ensure logs directory exists
const logsDir = path.join(process.cwd(), 'logs');
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir, { recursive: true });
}

const logFilePath = path.join(logsDir, 'registrations.log');
const usersFilePath = path.join(logsDir, 'users.json');

// Helper function to log to file
function logToFile(message: string) {
  const timestamp = new Date().toISOString();
  const logMessage = `[${timestamp}] ${message}\n`;
  fs.appendFileSync(logFilePath, logMessage, 'utf8');
  console.log(logMessage.trim());
}

// Simple file-based user storage (for development without database)
function getUsersFromFile(): any[] {
  try {
    if (fs.existsSync(usersFilePath)) {
      const data = fs.readFileSync(usersFilePath, 'utf8');
      return JSON.parse(data);
    }
    return [];
  } catch (error) {
    console.error('Error reading users file:', error);
    return [];
  }
}

function saveUsersToFile(users: any[]): boolean {
  try {
    fs.writeFileSync(usersFilePath, JSON.stringify(users, null, 2), 'utf8');
    return true;
  } catch (error) {
    console.error('Error saving users file:', error);
    return false;
  }
}

// Simple hash function (use bcrypt in production)
function hashPassword(password: string): string {
  return crypto.createHash('sha256').update(password).digest('hex');
}

router.post("/register", async (req, res) => {
  try {
    const { name, email, password } = req.body;

    // Validate input
    if (!name || !email || !password) {
      return res.status(400).json({ error: "Missing required fields" });
    }

    if (password.length < 8) {
      return res.status(400).json({ error: "Password must be at least 8 characters" });
    }

    // Check if user already exists (file-based)
    const users = getUsersFromFile();
    const existingUser = users.find((u: any) => u.email === email);

    if (existingUser) {
      logToFile(`Registration failed: Email already exists - ${email}`);
      return res.status(409).json({ error: "Email already registered" });
    }

    // Hash password
    const hashedPassword = hashPassword(password);

    // Create new user
    const newUser = {
      id: crypto.randomUUID(),
      name,
      email,
      password: hashedPassword,
      isApproved: false,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    // Save to file
    users.push(newUser);
    if (!saveUsersToFile(users)) {
      throw new Error("Failed to save user data");
    }

    // Log the registration
    logToFile(`New registration: ${name} (${email}) - ID: ${newUser.id} - Status: Pending Approval`);

    // Send success response
    res.status(201).json({
      message: "Registration successful",
      userId: newUser.id,
      email: email,
      status: "pending_approval",
      requiresApproval: true
    });

  } catch (error) {
    console.error("Registration error:", error);
    logToFile(`Registration error: ${error}`);
    res.status(500).json({ error: "Registration failed" });
  }
});

// Login endpoint
router.post("/login", async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ error: "Missing email or password" });
    }

    // Find user by email (file-based)
    const users = getUsersFromFile();
    const user = users.find((u: any) => u.email === email);

    if (!user) {
      logToFile(`Login failed: User not found - ${email}`);
      return res.status(401).json({ error: "Invalid credentials" });
    }

    // Check password
    const hashedPassword = hashPassword(password);
    if (user.password !== hashedPassword) {
      logToFile(`Login failed: Invalid password - ${email}`);
      return res.status(401).json({ error: "Invalid credentials" });
    }

    // Check if approved
    if (!user.isApproved) {
      logToFile(`Login failed: Account not approved - ${email}`);
      return res.status(403).json({ error: "Account pending approval" });
    }

    // Log successful login
    logToFile(`Login successful: ${user.name} (${email}) - ID: ${user.id}`);

    // Return success with user info (excluding password)
    const { password: _, ...userWithoutPassword } = user;
    res.status(200).json({
      message: "Login successful",
      user: userWithoutPassword
    });

  } catch (error) {
    console.error("Login error:", error);
    logToFile(`Login error: ${error}`);
    res.status(500).json({ error: "Login failed" });
  }
});

// Admin endpoint to approve users (for testing)
router.post("/admin/approve-user", async (req, res) => {
  try {
    const { userId } = req.body;

    if (!userId) {
      return res.status(400).json({ error: "Missing userId" });
    }

    const users = getUsersFromFile();
    const userIndex = users.findIndex((u: any) => u.id === userId);

    if (userIndex === -1) {
      return res.status(404).json({ error: "User not found" });
    }

    users[userIndex].isApproved = true;
    users[userIndex].updatedAt = new Date().toISOString();
    if (!saveUsersToFile(users)) {
      throw new Error("Failed to save user data");
    }

    logToFile(`User approved: ${users[userIndex].name} (${users[userIndex].email}) - ID: ${userId}`);

    res.status(200).json({
      message: "User approved successfully",
      user: {
        id: users[userIndex].id,
        name: users[userIndex].name,
        email: users[userIndex].email,
        isApproved: true
      }
    });

  } catch (error) {
    console.error("Approval error:", error);
    logToFile(`Approval error: ${error}`);
    res.status(500).json({ error: "Approval failed" });
  }
});

// Get all users (admin endpoint for testing)
router.get("/admin/users", async (req, res) => {
  try {
    const users = getUsersFromFile();
    const usersWithoutPasswords = users.map((u: any) => {
      const { password, ...userWithoutPassword } = u;
      return userWithoutPassword;
    });

    res.status(200).json({
      users: usersWithoutPasswords,
      count: usersWithoutPasswords.length
    });

  } catch (error) {
    console.error("Get users error:", error);
    res.status(500).json({ error: "Failed to get users" });
  }
});

export default router;