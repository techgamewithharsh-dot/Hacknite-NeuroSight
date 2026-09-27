import { db } from "../lib/db/src/index";
import { usersTable } from "../lib/db/src/schema";
import { sql } from "drizzle-orm";

async function setupDatabase() {
  console.log("Setting up database...");

  try {
    // Create users table
    await db.execute(sql`
      CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY DEFAULT gen_random_uuid(),
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        is_approved BOOLEAN DEFAULT false,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
      )
    `);

    console.log("✅ Database setup complete!");
    console.log("✅ Users table created successfully");

  } catch (error) {
    console.error("❌ Database setup failed:", error);
    process.exit(1);
  }
}

setupDatabase().then(() => {
  console.log("Setup finished");
  process.exit(0);
}).catch((error) => {
  console.error("Setup error:", error);
  process.exit(1);
});