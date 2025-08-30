import os

def generate_dashboard_page(output_dir="lib/pages"):
    """
    Generates a Flutter dashboard page file with AI/Data Science styled UI.
    The page includes cards for stats, a placeholder for charts, and
    a modern background gradient.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, "dashboard_page.dart")

    dashboard_code = """import 'package:flutter/material.dart';

class DashboardPage extends StatelessWidget {
  const DashboardPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFF1A237E), Color(0xFF0D47A1)], // Deep AI/Data Science gradient
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  "AI Dashboard",
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 20),

                // Stats Row
                Row(
                  children: [
                    Expanded(child: _buildStatCard("Users", "1200", Icons.people)),
                    const SizedBox(width: 12),
                    Expanded(child: _buildStatCard("Revenue", "\$8.5K", Icons.monetization_on)),
                  ],
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(child: _buildStatCard("Models Trained", "32", Icons.memory)),
                    const SizedBox(width: 12),
                    Expanded(child: _buildStatCard("Projects", "15", Icons.work)),
                  ],
                ),

                const SizedBox(height: 20),

                // Placeholder for chart
                Container(
                  height: 220,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: Colors.white24, width: 1),
                  ),
                  child: const Center(
                    child: Text(
                      "📊 Chart Placeholder",
                      style: TextStyle(color: Colors.white70, fontSize: 16),
                    ),
                  ),
                ),

                const SizedBox(height: 20),

                // Placeholder for activity list
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: const [
                      Text(
                        "Recent Activities",
                        style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      SizedBox(height: 10),
                      Text("✔️ New model deployed", style: TextStyle(color: Colors.white70)),
                      Text("✔️ User subscribed to Premium", style: TextStyle(color: Colors.white70)),
                      Text("✔️ Dataset uploaded", style: TextStyle(color: Colors.white70)),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  static Widget _buildStatCard(String title, String value, IconData icon) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white24, width: 1),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, color: Colors.white, size: 30),
          const SizedBox(height: 8),
          Text(value,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 20,
                fontWeight: FontWeight.bold,
              )),
          const SizedBox(height: 4),
          Text(title,
              style: const TextStyle(
                color: Colors.white70,
                fontSize: 14,
              )),
        ],
      ),
    );
  }
}
"""

    # Write the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(dashboard_code)

    print(f"✅ Dashboard page generated successfully at {file_path}")


def generate_dashboard(data: dict = None) -> dict:
    """
    Stub for FastAPI route usage.
    This prevents ImportError and can be extended later.
    """
    return {
        "status": "success",
        "message": "Dashboard generated",
        "data": data or {}
    }


if __name__ == "__main__":
    generate_dashboard_page()              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  "AI Dashboard",
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 20),

                // Stats Row
                Row(
                  children: [
                    Expanded(child: _buildStatCard("Users", "1200", Icons.people)),
                    const SizedBox(width: 12),
                    Expanded(child: _buildStatCard("Revenue", "\$8.5K", Icons.monetization_on)),
                  ],
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(child: _buildStatCard("Models Trained", "32", Icons.memory)),
                    const SizedBox(width: 12),
                    Expanded(child: _buildStatCard("Projects", "15", Icons.work)),
                  ],
                ),

                const SizedBox(height: 20),

                // Placeholder for chart
                Container(
                  height: 220,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: Colors.white24, width: 1),
                  ),
                  child: const Center(
                    child: Text(
                      "📊 Chart Placeholder",
                      style: TextStyle(color: Colors.white70, fontSize: 16),
                    ),
                  ),
                ),

                const SizedBox(height: 20),

                // Placeholder for activity list
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: const [
                      Text(
                        "Recent Activities",
                        style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      SizedBox(height: 10),
                      Text("✔️ New model deployed", style: TextStyle(color: Colors.white70)),
                      Text("✔️ User subscribed to Premium", style: TextStyle(color: Colors.white70)),
                      Text("✔️ Dataset uploaded", style: TextStyle(color: Colors.white70)),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  static Widget _buildStatCard(String title, String value, IconData icon) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white24, width: 1),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, color: Colors.white, size: 30),
          const SizedBox(height: 8),
          Text(value,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 20,
                fontWeight: FontWeight.bold,
              )),
          const SizedBox(height: 4),
          Text(title,
              style: const TextStyle(
                color: Colors.white70,
                fontSize: 14,
              )),
        ],
      ),
    );
  }
}
"""

    # Write the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(dashboard_code)

    print(f"✅ Dashboard page generated successfully at {file_path}")


def generate_dashboard(data: dict = None) -> dict:
    """
    Stub for FastAPI route usage.
    This prevents ImportError and can be extended later.
    """
    return {
        "status": "success",
        "message": "Dashboard generated",
        "data": data or {}
    }


if __name__ == "__main__":
    generate_dashboard_page()              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  "AI Dashboard",
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 20),

                // Stats Row
                Row(
                  children: [
                    Expanded(child: _buildStatCard("Users", "1200", Icons.people)),
                    const SizedBox(width: 12),
                    Expanded(child: _buildStatCard("Revenue", "\$8.5K", Icons.monetization_on)),
                  ],
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(child: _buildStatCard("Models Trained", "32", Icons.memory)),
                    const SizedBox(width: 12),
                    Expanded(child: _buildStatCard("Projects", "15", Icons.work)),
                  ],
                ),

                const SizedBox(height: 20),

                // Placeholder for chart
                Container(
                  height: 220,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: Colors.white24, width: 1),
                  ),
                  child: const Center(
                    child: Text(
                      "📊 Chart Placeholder",
                      style: TextStyle(color: Colors.white70, fontSize: 16),
                    ),
                  ),
                ),

                const SizedBox(height: 20),

                // Placeholder for activity list
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: const [
                      Text(
                        "Recent Activities",
                        style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      SizedBox(height: 10),
                      Text("✔️ New model deployed", style: TextStyle(color: Colors.white70)),
                      Text("✔️ User subscribed to Premium", style: TextStyle(color: Colors.white70)),
                      Text("✔️ Dataset uploaded", style: TextStyle(color: Colors.white70)),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  static Widget _buildStatCard(String title, String value, IconData icon) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white24, width: 1),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, color: Colors.white, size: 30),
          const SizedBox(height: 8),
          Text(value,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 20,
                fontWeight: FontWeight.bold,
              )),
          const SizedBox(height: 4),
          Text(title,
              style: const TextStyle(
                color: Colors.white70,
                fontSize: 14,
              )),
        ],
      ),
    );
  }
}
"""

    # Write the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(dashboard_code)

    print(f"✅ Dashboard page generated successfully at {file_path}")


if __name__ == "__main__":
    generate_dashboard_page()
