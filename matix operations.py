import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QTabWidget, QComboBox, QSizePolicy,
    QGroupBox, QScrollArea, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAction, QMenu, QStatusBar, QSplitter, QDialog
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor, QPalette

class MatrixInputDialog(QDialog):
    """Dialog for manual matrix input"""
    def __init__(self, rows, cols, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enter Matrix")
        self.setWindowIcon(QIcon(":matrix-icon"))
        self.setMinimumSize(500, 300)
        
        self.rows = rows
        self.cols = cols
        
        layout = QVBoxLayout()
        
        # Matrix input grid
        self.table = QTableWidget(rows, cols, self)
        self.table.setHorizontalHeaderLabels([f"Col {i+1}" for i in range(cols)])
        self.table.setVerticalHeaderLabels([f"Row {i+1}" for i in range(rows)])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Fill with zeros
        for i in range(rows):
            for j in range(cols):
                self.table.setItem(i, j, QTableWidgetItem("0"))
        
        layout.addWidget(self.table)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def get_matrix(self):
        """Return the matrix from user input"""
        matrix = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                item = self.table.item(i, j)
                if item and item.text():
                    try:
                        row.append(float(item.text()))
                    except ValueError:
                        row.append(0.0)
                else:
                    row.append(0.0)
            matrix.append(row)
        return np.array(matrix)

class MatrixOperationsApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Matrix Operations Tool")
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(self.style().standardIcon(getattr(self.style(), 'SP_FileDialogDetailedView')))
        
        # Initialize matrices
        self.matrices = {
            "Matrix A": None,
            "Matrix B": None,
            "Result": None
        }
        
        # Create UI
        self.init_ui()
        
        # Initialize status bar
        self.statusBar().showMessage("Ready")
    
    def init_ui(self):
        """Initialize the user interface"""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for left and right panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel for matrix input and operations
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Matrix selection and input section
        matrix_group = QGroupBox("Matrix Input")
        matrix_layout = QGridLayout(matrix_group)
        
        # Matrix selector
        matrix_layout.addWidget(QLabel("Select Matrix:"), 0, 0)
        self.matrix_selector = QComboBox()
        self.matrix_selector.addItems(["Matrix A", "Matrix B"])
        matrix_layout.addWidget(self.matrix_selector, 0, 1)
        
        # Matrix dimensions
        matrix_layout.addWidget(QLabel("Dimensions:"), 1, 0)
        self.dim_layout = QHBoxLayout()
        
        self.rows_input = QLineEdit("3")
        self.rows_input.setFixedWidth(50)
        self.dim_layout.addWidget(QLabel("Rows:"))
        self.dim_layout.addWidget(self.rows_input)
        
        self.cols_input = QLineEdit("3")
        self.cols_input.setFixedWidth(50)
        self.dim_layout.addWidget(QLabel("Cols:"))
        self.dim_layout.addWidget(self.cols_input)
        
        matrix_layout.addLayout(self.dim_layout, 1, 1)
        
        # Input buttons
        self.input_manual_btn = QPushButton("Manual Input")
        self.input_manual_btn.clicked.connect(self.manual_matrix_input)
        matrix_layout.addWidget(self.input_manual_btn, 2, 0, 1, 2)
        
        self.input_random_btn = QPushButton("Random Matrix")
        self.input_random_btn.clicked.connect(self.random_matrix)
        matrix_layout.addWidget(self.input_random_btn, 3, 0, 1, 2)
        
        self.input_file_btn = QPushButton("Load from File")
        self.input_file_btn.clicked.connect(self.load_matrix_from_file)
        matrix_layout.addWidget(self.input_file_btn, 4, 0, 1, 2)
        
        # Matrix display
        self.matrix_display = QTextEdit()
        self.matrix_display.setReadOnly(True)
        self.matrix_display.setFont(QFont("Courier New", 10))
        matrix_layout.addWidget(self.matrix_display, 5, 0, 1, 2)
        
        left_layout.addWidget(matrix_group)
        
        # Operations section
        operations_group = QGroupBox("Operations")
        operations_layout = QGridLayout(operations_group)
        
        # Operation buttons
        operations = [
            ("Addition", self.add_matrices),
            ("Subtraction", self.subtract_matrices),
            ("Multiplication", self.multiply_matrices),
            ("Transpose", self.transpose_matrix),
            ("Determinant", self.calculate_determinant),
            ("Inverse", self.inverse_matrix),
            ("Rank", self.calculate_rank),
            ("Eigenvalues", self.calculate_eigenvalues),
            ("Trace", self.calculate_trace),
            ("Identity", self.create_identity),
            ("Zero Matrix", self.create_zero_matrix),
            ("Clear", self.clear_matrices)
        ]
        
        row, col = 0, 0
        for name, func in operations:
            btn = QPushButton(name)
            btn.clicked.connect(func)
            btn.setMinimumHeight(40)
            operations_layout.addWidget(btn, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1
        
        left_layout.addWidget(operations_group)
        
        # Result section
        result_group = QGroupBox("Result")
        result_layout = QVBoxLayout(result_group)
        
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setFont(QFont("Courier New", 12))
        result_layout.addWidget(self.result_display)
        
        self.save_result_btn = QPushButton("Save Result as Matrix")
        self.save_result_btn.clicked.connect(self.save_result_as_matrix)
        result_layout.addWidget(self.save_result_btn)
        
        left_layout.addWidget(result_group)
        left_layout.setStretch(0, 2)
        left_layout.setStretch(1, 3)
        left_layout.setStretch(2, 3)
        
        # Right panel for history
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        history_group = QGroupBox("Operation History")
        history_layout = QVBoxLayout(history_group)
        
        self.history_display = QTextEdit()
        self.history_display.setReadOnly(True)
        self.history_display.setFont(QFont("Arial", 10))
        history_layout.addWidget(self.history_display)
        
        self.clear_history_btn = QPushButton("Clear History")
        self.clear_history_btn.clicked.connect(self.clear_history)
        history_layout.addWidget(self.clear_history_btn)
        
        right_layout.addWidget(history_group)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([800, 400])
        
        main_layout.addWidget(splitter)
        
        # Create menu bar
        self.create_menu()
        
        # Set initial matrix
        self.update_matrix_display()
    
    def create_menu(self):
        """Create the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)
        
        open_action = QAction("Open Matrix", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_matrix_from_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save Result", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_result)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        clear_action = QAction("Clear All", self)
        clear_action.setShortcut("Ctrl+Shift+C")
        clear_action.triggered.connect(self.clear_matrices)
        edit_menu.addAction(clear_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def manual_matrix_input(self):
        """Open dialog for manual matrix input"""
        try:
            rows = int(self.rows_input.text())
            cols = int(self.cols_input.text())
            if rows <= 0 or cols <= 0:
                raise ValueError("Dimensions must be positive integers")
            
            dialog = MatrixInputDialog(rows, cols, self)
            if dialog.exec_() == QDialog.Accepted:
                matrix = dialog.get_matrix()
                matrix_name = self.matrix_selector.currentText()
                self.matrices[matrix_name] = matrix
                self.update_matrix_display()
                self.add_to_history(f"Manual input for {matrix_name}: {rows}x{cols}")
                self.statusBar().showMessage(f"{matrix_name} updated successfully")
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", str(e))
    
    def random_matrix(self):
        """Generate a random matrix"""
        try:
            rows = int(self.rows_input.text())
            cols = int(self.cols_input.text())
            if rows <= 0 or cols <= 0:
                raise ValueError("Dimensions must be positive integers")
            
            matrix = np.random.rand(rows, cols) * 10 - 5  # Values between -5 and 5
            matrix = matrix.round(2)  # Round to 2 decimal places
            
            matrix_name = self.matrix_selector.currentText()
            self.matrices[matrix_name] = matrix
            self.update_matrix_display()
            self.add_to_history(f"Random matrix for {matrix_name}: {rows}x{cols}")
            self.statusBar().showMessage(f"{matrix_name} generated successfully")
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", str(e))
    
    def load_matrix_from_file(self):
        """Load matrix from a file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Matrix File", "", "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                matrix = np.loadtxt(file_path, delimiter=',')
                matrix_name = self.matrix_selector.currentText()
                self.matrices[matrix_name] = matrix
                self.update_matrix_display()
                self.add_to_history(f"Loaded {matrix_name} from {file_path}")
                self.statusBar().showMessage(f"{matrix_name} loaded successfully")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load matrix: {str(e)}")
    
    def save_result_as_matrix(self):
        """Save the result as a new matrix"""
        if self.matrices["Result"] is not None:
            self.matrices["Matrix B"] = self.matrices["Result"]
            self.matrix_selector.setCurrentIndex(1)  # Switch to Matrix B
            self.update_matrix_display()
            self.add_to_history("Result saved as Matrix B")
            self.statusBar().showMessage("Result saved as Matrix B")
        else:
            QMessageBox.warning(self, "Save Error", "No result to save")
    
    def save_result(self):
        """Save result to file"""
        if self.matrices["Result"] is None:
            QMessageBox.warning(self, "Save Error", "No result to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Result", "", "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                np.savetxt(file_path, self.matrices["Result"], delimiter=',', fmt='%.6f')
                self.add_to_history(f"Result saved to {file_path}")
                self.statusBar().showMessage(f"Result saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save result: {str(e)}")
    
    def update_matrix_display(self):
        """Update the matrix display"""
        matrix_name = self.matrix_selector.currentText()
        matrix = self.matrices[matrix_name]
        
        if matrix is None:
            self.matrix_display.setText("No matrix data")
            return
        
        rows, cols = matrix.shape
        display_text = f"{matrix_name} ({rows}x{cols}):\n\n"
        
        # Format matrix as string
        with np.printoptions(precision=3, suppress=True, linewidth=100):
            matrix_str = str(matrix)
        
        display_text += matrix_str
        self.matrix_display.setText(display_text)
    
    def format_matrix(self, matrix):
        """Format a matrix as a string"""
        if matrix is None:
            return "No matrix data"
        
        rows, cols = matrix.shape
        display_text = f"Matrix ({rows}x{cols}):\n\n"
        
        # Format matrix as string
        with np.printoptions(precision=4, suppress=True, linewidth=100):
            matrix_str = str(matrix)
        
        return display_text + matrix_str
    
    def add_to_history(self, message):
        """Add a message to the history display"""
        current_history = self.history_display.toPlainText()
        timestamp = QApplication.translate("MainWindow", "hh:mm:ss", None)
        new_entry = f"[{timestamp}] {message}"
        
        if current_history:
            self.history_display.setText(f"{new_entry}\n{current_history}")
        else:
            self.history_display.setText(new_entry)
    
    def clear_history(self):
        """Clear the history display"""
        self.history_display.clear()
    
    def clear_matrices(self):
        """Clear all matrices"""
        for key in self.matrices:
            self.matrices[key] = None
        self.update_matrix_display()
        self.result_display.clear()
        self.add_to_history("All matrices cleared")
        self.statusBar().showMessage("Matrices cleared")
    
    def new_project(self):
        """Start a new project"""
        self.clear_matrices()
        self.clear_history()
        self.statusBar().showMessage("New project created")
    
    def validate_matrices(self, require_square=False, require_two=False):
        """Validate matrices for operations"""
        matrix_name = self.matrix_selector.currentText()
        matrix = self.matrices[matrix_name]
        
        if matrix is None:
            QMessageBox.warning(self, "Operation Error", f"{matrix_name} is not defined")
            return False
        
        if require_square and matrix.shape[0] != matrix.shape[1]:
            QMessageBox.warning(self, "Operation Error", "Matrix must be square for this operation")
            return False
        
        if require_two:
            other_name = "Matrix B" if matrix_name == "Matrix A" else "Matrix A"
            other_matrix = self.matrices[other_name]
            if other_matrix is None:
                QMessageBox.warning(self, "Operation Error", f"{other_name} is not defined")
                return False
        
        return True
    
    # Matrix Operations
    def add_matrices(self):
        """Add two matrices"""
        if not self.validate_matrices(require_two=True):
            return
        
        matrix_a = self.matrices["Matrix A"]
        matrix_b = self.matrices["Matrix B"]
        
        try:
            result = matrix_a + matrix_b
            self.matrices["Result"] = result
            self.result_display.setText(self.format_matrix(result))
            self.add_to_history("Matrix addition: A + B")
            self.statusBar().showMessage("Addition completed successfully")
        except Exception as e:
            QMessageBox.critical(self, "Operation Error", f"Matrix addition failed: {str(e)}")
    
    def subtract_matrices(self):
        """Subtract two matrices"""
        if not self.validate_matrices(require_two=True):
            return
        
        matrix_a = self.matrices["Matrix A"]
        matrix_b = self.matrices["Matrix B"]
        
        try:
            result = matrix_a - matrix_b
            self.matrices["Result"] = result
            self.result_display.setText(self.format_matrix(result))
            self.add_to_history("Matrix subtraction: A - B")
            self.statusBar().showMessage("Subtraction completed successfully")
        except Exception as e:
            QMessageBox.critical(self, "Operation Error", f"Matrix subtraction failed: {str(e)}")
    
    def multiply_matrices(self):
        """Multiply two matrices"""
        if not self.validate_matrices(require_two=True):
            return
        
        matrix_a = self.matrices["Matrix A"]
        matrix_b = self.matrices["Matrix B"]
        
        try:
            result = matrix_a @ matrix_b
            self.matrices["Result"] = result
            self.result_display.setText(self.format_matrix(result))
            self.add_to_history("Matrix multiplication: A × B")
            self.statusBar().showMessage("Multiplication completed successfully")
        except Exception as e:
            QMessageBox.critical(self, "Operation Error", f"Matrix multiplication failed: {str(e)}")
    
    def transpose_matrix(self):
        """Transpose a matrix"""
        if not self.validate_matrices():
            return
        
        matrix_name = self.matrix_selector.currentText()
        matrix = self.matrices[matrix_name]
        
        try:
            result = matrix.T
            self.matrices["Result"] = result
            self.result_display.setText(self.format_matrix(result))
            self.add_to_history(f"Matrix transpose: {matrix_name}")
            self.statusBar().showMessage("Transpose completed successfully")
        except Exception as e:
            QMessageBox.critical(self, "Operation Error", f"Matrix transpose failed: {str(e)}")
    
    def calculate_determinant(self):
        """Calculate determinant of a matrix"""
        if not self.validate_matrices(require_square=True):
            return
        
        matrix_name = self.matrix_selector.currentText()
        matrix = self.matrices[matrix_name]
        
        try:
            det = np.linalg.det(matrix)
            result_text = f"Determinant of {matrix_name}:\n\n{det:.6f}"
            self.result_display.setText(result_text)
            self.matrices["Result"] = np.array([[det]])  # Save as 1x1 matrix
            self.add_to_history(f"Determinant calculation: {matrix_name}")
            self.statusBar().showMessage("Determinant calculated successfully")
        except Exception as e:
            QMessageBox.critical(self, "Operation Error", f"Determinant calculation failed: {str(e)}")
    
    def inverse_matrix(self):
        """Calculate inverse of a matrix"""
        if not self.validate_matrices(require_square=True):
            return
        
        matrix_name = self.matrix_selector.currentText()
        matrix = self.matrices[matrix_name]
        
        try:
            det = np.linalg.det(matrix)
            if abs(det) < 1e-10:  # Check if matrix is singular
                QMessageBox.warning(self, "Operation Error", "Matrix is singular (determinant is zero)")
                return
                
            result = np.linalg.inv(matrix)
            self.matrices["Result"] = result
            self.result_display.setText(self.format_matrix(result))
            self.add_to_history(f"Matrix inverse: {matrix_name}")
            self.statusBar().showMessage("Inverse calculated successfully")
        except Exception as e:
            QMessageBox.critical(self, "Operation Error", f"Matrix inversion failed: {str(e)}")
    
    def calculate_rank(self):
        """Calculate rank of a matrix"""
        if not self.validate_matrices():
            return
        
        matrix_name = self.matrix_selector.currentText()
        matrix = self.matrices[matrix_name]
        
        try:
            rank = np.linalg.matrix_rank(matrix)
            result_text = f"Rank of {matrix_name}:\n\n{rank}"
            self.result_display.setText(result_text)
            self.matrices["Result"] = np.array([[rank]])  # Save as 1x1 matrix
            self.add_to_history(f"Rank calculation: {matrix_name}")
            self.statusBar().showMessage("Rank calculated successfully")
        except Exception as e:
            QMessageBox.critical(self, "Operation Error", f"Rank calculation failed: {str(e)}")
    
    def calculate_eigenvalues(self):
        """Calculate eigenvalues of a matrix"""
        if not self.validate_matrices(require_square=True):
            return
        
        matrix_name = self.matrix_selector.currentText()
        matrix = self.matrices[matrix_name]
        
        try:
            eigenvalues = np.linalg.eigvals(matrix)
            result_text = f"Eigenvalues of {matrix_name}:\n\n"
            for i, val in enumerate(eigenvalues):
                result_text += f"λ{i+1}: {val:.6f}\n"
            
            self.result_display.setText(result_text)
            self.matrices["Result"] = np.diag(eigenvalues)  # Save as diagonal matrix
            self.add_to_history(f"Eigenvalue calculation: {matrix_name}")
            self.statusBar().showMessage("Eigenvalues calculated successfully")
        except Exception as e:
            QMessageBox.critical(self, "Operation Error", f"Eigenvalue calculation failed: {str(e)}")
    
    def calculate_trace(self):
        """Calculate trace of a matrix"""
        if not self.validate_matrices(require_square=True):
            return
        
        matrix_name = self.matrix_selector.currentText()
        matrix = self.matrices[matrix_name]
        
        try:
            trace = np.trace(matrix)
            result_text = f"Trace of {matrix_name}:\n\n{trace:.6f}"
            self.result_display.setText(result_text)
            self.matrices["Result"] = np.array([[trace]])  # Save as 1x1 matrix
            self.add_to_history(f"Trace calculation: {matrix_name}")
            self.statusBar().showMessage("Trace calculated successfully")
        except Exception as e:
            QMessageBox.critical(self, "Operation Error", f"Trace calculation failed: {str(e)}")
    
    def create_identity(self):
        """Create an identity matrix"""
        try:
            size = int(self.rows_input.text())
            if size <= 0:
                raise ValueError("Size must be positive integer")
            
            matrix = np.eye(size)
            matrix_name = self.matrix_selector.currentText()
            self.matrices[matrix_name] = matrix
            self.update_matrix_display()
            self.add_to_history(f"Created identity matrix for {matrix_name}: {size}x{size}")
            self.statusBar().showMessage("Identity matrix created")
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", str(e))
    
    def create_zero_matrix(self):
        """Create a zero matrix"""
        try:
            rows = int(self.rows_input.text())
            cols = int(self.cols_input.text())
            if rows <= 0 or cols <= 0:
                raise ValueError("Dimensions must be positive integers")
            
            matrix = np.zeros((rows, cols))
            matrix_name = self.matrix_selector.currentText()
            self.matrices[matrix_name] = matrix
            self.update_matrix_display()
            self.add_to_history(f"Created zero matrix for {matrix_name}: {rows}x{cols}")
            self.statusBar().showMessage("Zero matrix created")
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", str(e))
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h2>Matrix Operations Tool</h2>
        <p>Version 1.0</p>
        <p>This application provides a comprehensive set of matrix operations using NumPy.</p>
        <p>Features include:</p>
        <ul>
            <li>Matrix addition, subtraction, and multiplication</li>
            <li>Matrix transpose and inverse</li>
            <li>Determinant, rank, and trace calculation</li>
            <li>Eigenvalue computation</li>
            <li>Manual and random matrix creation</li>
            <li>File import/export</li>
            <li>Operation history</li>
        </ul>
        <p>Created with PyQt5 and NumPy</p>
        """
        QMessageBox.about(self, "About Matrix Operations Tool", about_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and customize palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    # Set application font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MatrixOperationsApp()
    window.show()
    sys.exit(app.exec_())