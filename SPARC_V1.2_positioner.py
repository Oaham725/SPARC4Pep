#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: HAO MA with vibe coding by Trae
# Date: 2026-01-21

"""
SPARC v1.2 for Multi-chain PDB Raman calculation (TYR/PHE) with:
1) Custom lightweight PDB parser (original, unchanged in behavior)
2) Robust rotation matrices from two-vector-pair frame alignment (no ambiguous multi-step Rodrigues)
3) Chain-level residue->rotation cache (removes O(N^2) redundancy)
4) Excel-driven mode tensors + residue1/2/3 averaging
5) Formatted TXT output (same header convention as your SaveFiles)

You can copy-paste and run as a single script.

SPARC v1.2 多链PDB拉曼光谱计算（TYR/PHE），具有以下特点：
1) 自定义轻量级PDB解析器（原始代码，行为不变）
2) 基于双向量对框架对齐的稳健旋转矩阵（无歧义的多步骤Rodrigues方法）
3) 链级残基->旋转缓存（消除O(N^2)冗余）
4) Excel驱动的模式张量 + residue1/2/3平均
5) 格式化TXT输出（与原始SaveFiles相同的头部约定）

您可以复制粘贴并作为单个脚本运行。
"""

import os
import math
import numpy as np
import pandas as pd


# =========================================================
#  PDB lightweight object model (same as SPARC code)
# PDB轻量级对象模型（与原始代码相同）
# =========================================================
class Atom:
    """表示PDB文件中的一个原子
    Represents an atom in a PDB file
    """
    def __init__(self, name, coord, fullname=None, altloc=' ', serial_number=None):
        """初始化原子对象
        Initialize an Atom object
        
        参数 Parameters:
        - name: 原子名称 Atom name
        - coord: 原子坐标 Atom coordinates
        - fullname: 原子全名 Full atom name (optional)
        - altloc: 替代位置标识符 Alternate location identifier (optional)
        - serial_number: 原子序列号 Atom serial number (optional)
        """
        self.name = name
        self.fullname = fullname if fullname else name
        self.coord = coord
        self.altloc = altloc
        self.serial_number = serial_number
        self.parent = None

    def get_name(self):
        """获取原子名称
        Get the atom name
        
        返回 Returns:
        - 原子名称 Atom name
        """
        return self.name

    def get_coord(self):
        """获取原子坐标
        Get the atom coordinates
        
        返回 Returns:
        - 原子坐标 Atom coordinates
        """
        return self.coord

    def get_vector(self):
        """获取原子坐标向量
        Get the atom coordinate vector
        
        返回 Returns:
        - 原子坐标向量 Atom coordinate vector
        """
        return self.coord


class Residue:
    """表示PDB文件中的一个残基
    Represents a residue in a PDB file
    """
    def __init__(self, name, res_id, segid=''):
        """初始化残基对象
        Initialize a Residue object
        
        参数 Parameters:
        - name: 残基名称 Residue name
        - res_id: 残基ID Residue ID
        - segid: 段标识符 Segment identifier (optional)
        """
        self.name = name
        self.id = res_id
        self.segid = segid
        self.atoms = []
        self.parent = None

    def get_resname(self):
        """获取残基名称
        Get the residue name
        
        返回 Returns:
        - 残基名称 Residue name
        """
        return self.name

    def get_atoms(self):
        """获取残基中的所有原子
        Get all atoms in the residue
        
        返回 Returns:
        - 原子列表 List of atoms
        """
        return self.atoms

    def add_atom(self, atom):
        """添加一个原子到残基中
        Add an atom to the residue
        
        参数 Parameters:
        - atom: 要添加的原子 Atom to add
        """
        self.atoms.append(atom)
        atom.parent = self

    def has_id(self, atom_name):
        """检查残基是否包含指定名称的原子
        Check if the residue contains an atom with the specified name
        
        参数 Parameters:
        - atom_name: 原子名称 Atom name to check
        
        返回 Returns:
        - 如果包含指定名称的原子则返回True，否则返回False
        True if the residue contains an atom with the specified name, False otherwise
        """
        for atom in self.atoms:
            if atom.name == atom_name:
                return True
        return False


class Chain:
    """表示PDB文件中的一个链
    Represents a chain in a PDB file
    """
    def __init__(self, chain_id):
        """初始化链对象
        Initialize a Chain object
        
        参数 Parameters:
        - chain_id: 链ID Chain ID
        """
        self.id = chain_id
        self.residues = []
        self.parent = None

    def get_list(self):
        """获取链中的所有残基
        Get all residues in the chain
        
        返回 Returns:
        - 残基列表 List of residues
        """
        return self.residues

    def add_residue(self, residue):
        """添加一个残基到链中
        Add a residue to the chain
        
        参数 Parameters:
        - residue: 要添加的残基 Residue to add
        """
        self.residues.append(residue)
        residue.parent = self

    def has_id(self, res_id):
        """检查链是否包含指定ID的残基
        Check if the chain contains a residue with the specified ID
        
        参数 Parameters:
        - res_id: 残基ID Residue ID to check
        
        返回 Returns:
        - 如果包含指定ID的残基则返回True，否则返回False
        True if the chain contains a residue with the specified ID, False otherwise
        """
        for residue in self.residues:
            if residue.id == res_id:
                return True
        return False


class Model:
    """表示PDB文件中的一个模型
    Represents a model in a PDB file
    """
    def __init__(self, model_id, serial_num=None):
        """初始化模型对象
        Initialize a Model object
        
        参数 Parameters:
        - model_id: 模型ID Model ID
        - serial_num: 模型序列号 Model serial number (optional)
        """
        self.id = model_id
        self.serial_num = serial_num
        self.chains = {}
        self.parent = None

    def __getitem__(self, chain_id):
        """通过链ID获取链
        Get a chain by its ID
        
        参数 Parameters:
        - chain_id: 链ID Chain ID
        
        返回 Returns:
        - 链对象 Chain object if found, None otherwise
        """
        return self.chains.get(chain_id)

    def add_chain(self, chain):
        """添加一个链到模型中
        Add a chain to the model
        
        参数 Parameters:
        - chain: 要添加的链 Chain to add
        """
        self.chains[chain.id] = chain
        chain.parent = self

    def has_id(self, chain_id):
        """检查模型是否包含指定ID的链
        Check if the model contains a chain with the specified ID
        
        参数 Parameters:
        - chain_id: 链ID Chain ID to check
        
        返回 Returns:
        - 如果包含指定ID的链则返回True，否则返回False
        True if the model contains a chain with the specified ID, False otherwise
        """
        return chain_id in self.chains


class Structure:
    """表示PDB文件中的整个结构
    Represents the entire structure in a PDB file
    """
    def __init__(self, structure_id):
        """初始化结构对象
        Initialize a Structure object
        
        参数 Parameters:
        - structure_id: 结构ID Structure ID
        """
        self.id = structure_id
        self.models = []
        self.header = {}

    def __getitem__(self, model_id):
        """通过模型ID获取模型
        Get a model by its ID
        
        参数 Parameters:
        - model_id: 模型ID Model ID
        
        返回 Returns:
        - 模型对象 Model object
        """
        return self.models[model_id]

    def add_model(self, model):
        """添加一个模型到结构中
        Add a model to the structure
        
        参数 Parameters:
        - model: 要添加的模型 Model to add
        """
        self.models.append(model)
        model.parent = self


class PDBConstructionException(Exception):
    """PDB构建异常
    Exception raised during PDB construction
    """
    pass


class PDBParser:
    """解析PDB文件并构建结构对象
    Parses PDB files and builds structure objects
    """
    def __init__(self, PERMISSIVE=True, QUIET=False):
        """初始化PDB解析器
        Initialize a PDBParser object
        
        参数 Parameters:
        - PERMISSIVE: 是否允许宽松解析 Whether to allow permissive parsing
        - QUIET: 是否安静模式 Whether to run in quiet mode
        """
        self.permissive = bool(PERMISSIVE)
        self.quiet = bool(QUIET)
        self.line_counter = 0

    def get_structure(self, structure_id, filename):
        """从PDB文件获取结构
        Get structure from a PDB file
        
        参数 Parameters:
        - structure_id: 结构ID Structure ID
        - filename: PDB文件路径 PDB file path
        
        返回 Returns:
        - 结构对象 Structure object
        
        异常 Raises:
        - PDBConstructionException: 如果解析过程中出错 If an error occurs during parsing
        """
        structure = Structure(structure_id)

        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            raise PDBConstructionException(f"无法打开文件 {filename}: {e}")

        if not lines:
            raise PDBConstructionException("空文件")

        current_model = None
        current_chain = None
        current_residue = None
        current_residue_id = None
        current_resname = None
        current_chain_id = None
        model_open = False
        current_model_id = 0

        allowed_records = {
            "ATOM  ", "HETATM", "MODEL ", "ENDMDL", "TER   ",
            "ANISOU", "SIGATM", "SIGUIJ", "MASTER",
        }

        for line in lines:
            self.line_counter += 1
            line = line.rstrip('\n')

            if not line.strip():
                continue

            if len(line) < 6:
                if not self.quiet:
                    print(f"警告: 行 {self.line_counter} 太短，跳过")
                continue

            record_type = line[0:6]

            if record_type == "MODEL ":
                try:
                    serial_num = int(line[10:14].strip())
                except Exception:
                    if not self.quiet:
                        print(f"警告: 模型序列号无效，行 {self.line_counter}")
                    serial_num = None

                current_model = Model(current_model_id, serial_num)
                structure.add_model(current_model)
                current_model_id += 1
                model_open = True
                current_chain = None
                current_residue = None
                current_residue_id = None
                current_chain_id = None
                current_resname = None

            elif record_type in ("ATOM  ", "HETATM"):
                if not current_model:
                    current_model = Model(0)
                    structure.add_model(current_model)
                    model_open = True

                try:
                    fullname = line[12:16]
                    split_list = fullname.split()
                    if len(split_list) != 1:
                        atom_name = fullname
                    else:
                        atom_name = split_list[0]

                    altloc = line[16]
                    res_name = line[17:20].strip()
                    chain_id = line[21:22].strip()

                    try:
                        res_num = int(line[22:26].split()[0])
                    except Exception:
                        if self.permissive:
                            if not self.quiet:
                                print(f"警告: 残基序号无效，行 {self.line_counter}")
                            continue
                        else:
                            raise PDBConstructionException(f"残基序号无效，行 {self.line_counter}")

                    icode = line[26]

                    if record_type == "HETATM":
                        if res_name in ("HOH", "WAT"):
                            hetero_flag = "W"
                        else:
                            hetero_flag = "H"
                    else:
                        hetero_flag = " "

                    residue_id = (hetero_flag, res_num, icode)

                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                    except Exception:
                        if self.permissive:
                            if not self.quiet:
                                print(f"警告: 坐标无效，行 {self.line_counter}")
                            continue
                        else:
                            raise PDBConstructionException(f"坐标无效，行 {self.line_counter}")

                    atom = Atom(atom_name, np.array([x, y, z], dtype=float), fullname, altloc)

                    if not current_chain or current_chain_id != chain_id:
                        if current_model.has_id(chain_id):
                            current_chain = current_model[chain_id]
                            if not self.quiet:
                                print(f"警告: 链 {chain_id} 不连续，行 {self.line_counter}")
                        else:
                            current_chain = Chain(chain_id)
                            current_model.add_chain(current_chain)

                        current_chain_id = chain_id
                        current_residue = None
                        current_residue_id = None
                        current_resname = None

                    if (not current_residue) or (current_residue_id != residue_id) or (current_resname != res_name):
                        current_residue = Residue(res_name, residue_id)
                        current_chain.add_residue(current_residue)
                        current_residue_id = residue_id
                        current_resname = res_name

                    current_residue.add_atom(atom)

                except Exception as e:
                    if self.permissive:
                        if not self.quiet:
                            print(f"警告: 解析原子时出错，行 {self.line_counter}: {e}")
                        continue
                    else:
                        raise PDBConstructionException(f"解析原子时出错，行 {self.line_counter}: {e}")

            elif record_type == "ENDMDL":
                current_model = None
                model_open = False
                current_chain = None
                current_residue = None
                current_residue_id = None
                current_chain_id = None
                current_resname = None

            elif record_type == "TER   ":
                current_chain = None
                current_residue = None
                current_residue_id = None
                current_resname = None

            elif record_type not in allowed_records:
                if not self.quiet:
                    print(f"警告: 忽略未识别的记录类型 '{record_type}'，行 {self.line_counter}")

        return structure


# =========================================================
#  Robust rotation matrix from two-vector-pair frames
# 基于双向量对框架的稳健旋转矩阵
# =========================================================
class RM:
    """旋转矩阵类
    Rotation matrix class
    """
    EPS = 1e-12  # 极小值，用于避免除零错误

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        """归一化向量
        Normalize a vector
        
        参数 Parameters:
        - v: 输入向量 Input vector
        
        返回 Returns:
        - 归一化后的向量 Normalized vector
        
        异常 Raises:
        - ValueError: 如果向量长度为零 If the vector has zero length
        """
        v = np.asarray(v, dtype=float).reshape(3)
        n = np.linalg.norm(v)
        if n < RM.EPS:
            raise ValueError("Zero-length vector encountered while normalizing.")
        return v / n

    @staticmethod
    def _build_basis(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        使用两个非共线向量构建右手正交基B=[e1 e2 e3]
        基向量作为列。
        
        Use two non-collinear vectors to build a right-handed orthonormal basis B=[e1 e2 e3]
        with basis vectors as columns.
        
        参数 Parameters:
        - v1: 第一个向量 First vector
        - v2: 第二个向量 Second vector
        
        返回 Returns:
        - 正交基矩阵 Orthonormal basis matrix
        
        异常 Raises:
        - ValueError: 如果两个向量几乎共线 If the two vectors are nearly collinear
        """
        e1 = RM._normalize(v1)
        tmp = np.cross(e1, v2)
        n3 = np.linalg.norm(tmp)
        if n3 < RM.EPS:
            raise ValueError("v1 and v2 are nearly collinear; cannot define a unique frame.")
        e3 = tmp / n3
        e2 = np.cross(e3, e1)
        return np.column_stack([e1, e2, e3])

    @staticmethod
    def rotation_from_two_vector_pairs(std_v1, std_v2, tgt_v1, tgt_v2) -> np.ndarray:
        """
        返回旋转矩阵R，使得：B_tgt ≈ R * B_std（列是基向量）
        即 R = B_tgt * B_std^T
        
        Return R such that: B_tgt ≈ R * B_std  (columns are basis vectors)
        i.e. R = B_tgt * B_std^T
        
        参数 Parameters:
        - std_v1: 标准框架的第一个向量 First vector of the standard frame
        - std_v2: 标准框架的第二个向量 Second vector of the standard frame
        - tgt_v1: 目标框架的第一个向量 First vector of the target frame
        - tgt_v2: 目标框架的第二个向量 Second vector of the target frame
        
        返回 Returns:
        - 旋转矩阵 Rotation matrix
        """
        B_std = RM._build_basis(std_v1, std_v2)
        B_tgt = RM._build_basis(tgt_v1, tgt_v2)
        R = B_tgt @ B_std.T

        # 通过SVD投影到最近的正交旋转矩阵（SO(3)）
        # Project to nearest proper rotation matrix (SO(3)) by SVD
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return R


# =========================================================
#  Excel matrix handling & Raman intensity computation
# =========================================================
def _as_int_or_none(x):
    if x is None:
        return None
    try:
        if isinstance(x, float) and np.isnan(x):
            return None
        return int(x)
    except Exception:
        return None


class Matrix:
    # Simple cache to avoid re-reading Excel for same residue type
    _excel_cache = {}

    @staticmethod
    def SymMatrix(data: np.ndarray) -> np.ndarray:
        """
        将6列向量转换为对称矩阵
        
        参数 Parameters:
        - data: (N,6) 按顺序 [xx, xy, xz, yy, yz, zz]
        
        返回 Returns:
        - (N,3,3) 对称张量
        
        Convert 6-column vectors to symmetric matrices
        
        Parameters:
        - data: (N,6) in order [xx, xy, xz, yy, yz, zz]
        
        Returns:
        - (N,3,3) symmetric tensors
        """
        data = np.asarray(data, dtype=float)
        if data.shape[1] < 6:
            raise ValueError("Matrix data must have at least 6 columns.")
        N = data.shape[0]
        out = np.zeros((N, 3, 3), dtype=float)
        out[:, 0, 0] = data[:, 0]
        out[:, 0, 1] = data[:, 1]
        out[:, 1, 0] = data[:, 1]
        out[:, 0, 2] = data[:, 2]
        out[:, 2, 0] = data[:, 2]
        out[:, 1, 1] = data[:, 3]
        out[:, 1, 2] = data[:, 4]
        out[:, 2, 1] = data[:, 4]
        out[:, 2, 2] = data[:, 5]
        return out

    @staticmethod
    def LoadMartix(FilePath):
        """
        从Excel文件加载矩阵数据
        
        Excel文件预期列：
          - wavenumber, label, width
          - 可选：residue1, residue2, residue3
          - 前6列是张量分量 [xx, xy, xz, yy, yz, zz]
        
        参数 Parameters:
        - FilePath: Excel文件路径 Excel file path
        
        返回 Returns:
        - (MatrixSet, Label, WaveNumber, Width, residue1, residue2, residue3)
        
        Load matrix data from Excel file
        
        Excel expected columns:
          - wavenumber, label, width
          - optional: residue1, residue2, residue3
          - first 6 columns are tensor components [xx, xy, xz, yy, yz, zz]
        
        Parameters:
        - FilePath: Excel file path
        
        Returns:
        - (MatrixSet, Label, WaveNumber, Width, residue1, residue2, residue3)
        """
        FilePath = os.path.abspath(FilePath)
        if FilePath in Matrix._excel_cache:
            return Matrix._excel_cache[FilePath]

        df = pd.read_excel(FilePath)

        # required columns
        WaveNumber = df["wavenumber"].to_numpy(dtype=float)
        Label = df["label"].to_numpy()
        Width = df["width"].to_numpy(dtype=float)

        residue1 = df.get('residue1', pd.Series([None] * len(df))).to_numpy()
        residue2 = df.get('residue2', pd.Series([None] * len(df))).to_numpy()
        residue3 = df.get('residue3', pd.Series([None] * len(df))).to_numpy()

        residue1 = np.array([_as_int_or_none(v) for v in residue1], dtype=object)
        residue2 = np.array([_as_int_or_none(v) for v in residue2], dtype=object)
        residue3 = np.array([_as_int_or_none(v) for v in residue3], dtype=object)

        # first 6 columns are tensor data
        matrix_data = df.iloc[:, :6].to_numpy(dtype=float)
        MatrixSet = Matrix.SymMatrix(matrix_data)

        Matrix._excel_cache[FilePath] = (MatrixSet, Label, WaveNumber, Width, residue1, residue2, residue3)
        return MatrixSet, Label, WaveNumber, Width, residue1, residue2, residue3

    @staticmethod
    def ProcessMartix(MatrixSet, WaveNumber, Label, Z, T_by_label, residue1=None, residue2=None, residue3=None, residue_T_dict=None, chain_id=None, log_print=None):
        """
        处理矩阵数据并计算拉曼光谱强度
        
        参数 Parameters:
        - MatrixSet: (N,3,3) 局部极化率张量（来自Excel）
        - Label: (N,) 模式标签 (1..5)
        - T_by_label: 字典 {1:R1,2:R2,3:R3,...} 当前残基（单残基情况）
        - residue_T_dict: 字典 {resnum:[R1,R2,R3,...]} 用于residue1/2/3平均情况
        - Z: Z坐标 Z-coordinate
        - chain_id: 链ID Chain ID
        - log_print: 日志打印函数 Log print function
        
        强度模型（与原始保持一致）：
          s = (ez^T * A_global * ez) * 100
          I = s^2 * post_scale
        其中 ez = [0,0,exp(-0.009*Z^2)]
        
        Process matrix data and compute Raman spectrum intensity
        
        Parameters:
        - MatrixSet: (N,3,3) local polarizability tensors (from Excel)
        - Label: (N,) mode label (1..5)
        - T_by_label: dict {1:R1,2:R2,3:R3,...} for current residue (if single-residue case)
        - residue_T_dict: dict {resnum:[R1,R2,R3,...]} for residue1/2/3 averaging case
        - Z: Z-coordinate
        - chain_id: Chain ID
        - log_print: Log print function

        Intensity model (kept consistent with original):
          s = (ez^T * A_global * ez) * 100
          I = s^2 * post_scale
        where ez = [0,0,exp(-0.009*Z^2)]
        """
        MatrixSet = np.asarray(MatrixSet, dtype=float)
        WaveNumber = np.asarray(WaveNumber, dtype=float)
        Label = np.asarray(Label)

        #ez = np.array([0.0, 0.0, np.exp(-0.009 * (Z ** 2))], dtype=float)
        ez = np.array([0.0, 0.0, 1], dtype=float)
        scale = 1

        # 默认的log_print函数
        # Default log_print function
        def default_log_print(message):
            print(message)
        
        log = log_print or default_log_print
        
        def intensity_from_R(A_local, R):
            # tensor rotation: A_global = R * A_local * R^T
            A_g = R @ A_local @ R.T
            s = (ez @ A_g @ ez) * scale
            return float(s * s)

        def pick_R(label, resnum=None, wavenumber=None):
            try:
                lab = int(label)
            except Exception:
                return None

            if resnum is not None and residue_T_dict is not None:
                mats = residue_T_dict.get(resnum)
                if mats is None:
                    return None
                idx = lab - 1
                if idx < 0 or idx >= len(mats):
                    return None
                # 打印每次调用的旋转矩阵对应的残基、label和波数
                log(f"=== 调用旋转矩阵 ===")
                log(f"链: {chain_id}, 残基: {resnum}, Label: {lab}, 波数: {wavenumber}")
                log(f"  使用矩阵: T{lab}")
                return mats[idx]
            # 打印每次调用的旋转矩阵对应的label和波数
            log(f"=== 调用旋转矩阵 ===")
            log(f"链: {chain_id}, Label: {lab}, 波数: {wavenumber}")
            log(f"  使用矩阵: T{lab}")
            return T_by_label.get(lab)

        #C_function = 0.1
        post_scale = 1 # (C_function if Z < 20 else 1000.0)

        S_Set, WaveSet = [], []
        for i in range(MatrixSet.shape[0]):
            A = MatrixSet[i]

            # Multi-residue averaging if residue columns exist AND dict provided
            res_list = []
            if residue1 is not None and residue2 is not None and residue3 is not None and residue_T_dict is not None:
                for r in (residue1[i], residue2[i], residue3[i]):
                    r = _as_int_or_none(r)
                    if r is not None:
                        res_list.append(r)

            if len(res_list) > 0:
                vals = []
                for r in res_list:
                    R = pick_R(Label[i], resnum=r, wavenumber=WaveNumber[i])
                    if R is None:
                        continue
                    vals.append(intensity_from_R(A, R))
                if len(vals) == 0:
                    continue
                S_Set.append(float(np.mean(vals)) * post_scale)
                WaveSet.append(float(WaveNumber[i]))
            else:
                R = pick_R(Label[i], resnum=None, wavenumber=WaveNumber[i])
                if R is None:
                    continue
                S_Set.append(intensity_from_R(A, R) * post_scale)
                WaveSet.append(float(WaveNumber[i]))

        return S_Set, WaveSet

    @staticmethod
    def SaveData(DataSet, WaveNumber, Width, residue_id=None):
        """
        保存数据为数组格式
        
        参数 Parameters:
        - DataSet: 数据集 Data set
        - WaveNumber: 波数 Wavenumber
        - Width: 宽度 Width
        - residue_id: 残基ID（可选） Residue ID (optional)
        
        返回 Returns:
        - 连接后的数据数组 Concatenated data array
        
        Save data as array format
        
        Parameters:
        - DataSet: Data set
        - WaveNumber: Wavenumber
        - Width: Width
        - residue_id: Residue ID (optional)
        
        Returns:
        - Concatenated data array
        """
        DataSet = np.asarray(DataSet, dtype=float).reshape(-1, 1)
        WaveNumber = np.asarray(WaveNumber, dtype=float).reshape(-1, 1)
        Width = np.asarray(Width, dtype=float).reshape(-1, 1)[:DataSet.shape[0]]

        if residue_id is not None:
            residue_id_col = np.full((DataSet.shape[0], 1), int(residue_id), dtype=int)
            temp = np.concatenate((WaveNumber, DataSet, Width, residue_id_col), axis=1)
        else:
            temp = np.concatenate((WaveNumber, DataSet, Width), axis=1)
        return temp


def SaveFiles(DataSet, SavePath, input_file=None):
    """
    保存数据到文件
    
    参数 Parameters:
    - DataSet: 要保存的数据集 Data set to save
    - SavePath: 保存路径 Save path
    - input_file: 输入文件（可选） Input file (optional)
    """
    import datetime
    os.makedirs(os.path.dirname(os.path.abspath(SavePath)), exist_ok=True)
    np.savetxt(SavePath, DataSet, fmt='%.20f')
    with open(SavePath, "r+", encoding="utf-8") as f:
        content = f.read()
        f.seek(0, 0)
        if DataSet.shape[1] == 4:
            header = f"{DataSet.shape[0]} 3\n"
        else:
            header = f"{DataSet.shape[0]} 2\n"
        f.write(header + content)


# =========================================================
#  Protein pipeline
# =========================================================
def compute_T_mats_for_residue(residue: Residue, chain_id: str, std_vectors: dict, log_print=None):
    """
    为TYR/PHE残基计算3个旋转矩阵：
      R1: 苯环局部框架，从 (phStd1, phStd2) -> (v_ph1, v_ph2)
      R2: 环与CH2框架，从 (phStd1, CH2td1) -> (v_ph1, v_CH2)
      R3: 主链框架，从 (CCO_Std, CN_Std) -> (v_CCO, v_CN)

    返回：
      ( [R1,R2,R3], CA_vector )
    
    Compute 3 rotation matrices for a TYR/PHE residue:
      R1: phenyl ring local frame from (phStd1, phStd2) -> (v_ph1, v_ph2)
      R2: ring vs CH2 frame from (phStd1, CH2td1) -> (v_ph1, v_CH2)
      R3: backbone frame from (CCO_Std, CN_Std) -> (v_CCO, v_CN)

    Parameters:
    - residue: Residue object
    - chain_id: chain ID
    - std_vectors: standard vectors for rotation
    - log_print: logging function

    Returns:
      ( [R1,R2,R3], CA_vector )
    """
    # 默认的log_print函数
    # Default log_print function
    def default_log_print(message):
        print(message)
    
    log = log_print or default_log_print
    
    atom_dict = {atom.name: atom for atom in residue.get_atoms()}
    need = ['N', 'CA', 'CB', 'CG', 'CD1', 'C']
    if not all(a in atom_dict for a in need):
        return None

    CA = atom_dict['CA'].get_vector()
    CB = atom_dict['CB'].get_vector()
    CG = atom_dict['CG'].get_vector()
    CD1 = atom_dict['CD1'].get_vector()
    N = atom_dict['N'].get_vector()
    C = atom_dict['C'].get_vector()

    v_CCO = (C - CA)[:3]
    v_CN = (N - CA)[:3]
    v_CH2 = (CB - CA)[:3]
    v_ph1 = (CG - CB)[:3]
    v_ph2 = (CG - CD1)[:3]

    res_name = residue.get_resname()
    res_num = residue.id[1]

    log(f"=== 计算残基旋转矩阵 ===")
    log(f"链: {chain_id}, 残基: {res_name} {res_num}")
    log(f"  CA: {CA[:3]}")
    log(f"  CB: {CB[:3]}")
    log(f"  CG: {CG[:3]}")
    log(f"  CD1: {CD1[:3]}")
    log(f"  N: {N[:3]}")
    log(f"  C: {C[:3]}")
    log(f"  向量 v_CCO: {v_CCO}")
    log(f"  向量 v_CN: {v_CN}")
    log(f"  向量 v_CH2: {v_CH2}")
    log(f"  向量 v_ph1: {v_ph1}")
    log(f"  向量 v_ph2: {v_ph2}")

    key_full = (res_name, chain_id, res_num)
    key_no_chain = (res_name, res_num)

    if key_full in std_vectors:
        std = std_vectors[key_full]
    elif key_no_chain in std_vectors:
        std = std_vectors[key_no_chain]
    else:
        return None

    phStd1 = np.array(std['phStd1'], dtype=float)
    phStd2 = np.array(std['phStd2'], dtype=float)
    CH2td1 = np.array(std['CH2td1'], dtype=float)
    CCO_Std = np.array(std['CCO_Std'], dtype=float)
    CN_Std = np.array(std['CN_Std'], dtype=float)

    log(f"  标准向量 phStd1: {phStd1}")
    log(f"  标准向量 phStd2: {phStd2}")
    log(f"  标准向量 CH2td1: {CH2td1}")
    log(f"  标准向量 CCO_Std: {CCO_Std}")
    log(f"  标准向量 CN_Std: {CN_Std}")

    R1 = RM.rotation_from_two_vector_pairs(phStd1, phStd2, v_ph1, v_ph2)
    R2 = RM.rotation_from_two_vector_pairs(phStd1, CH2td1, v_ph1, v_CH2)
    R3 = RM.rotation_from_two_vector_pairs(CCO_Std, CN_Std, v_CCO, v_CN)

    log(f"  计算得到 R1 矩阵:")
    log(f"  {R1[0]}")
    log(f"  {R1[1]}")
    log(f"  {R1[2]}")
    log(f"  计算得到 R2 矩阵:")
    log(f"  {R2[0]}")
    log(f"  {R2[1]}")
    log(f"  {R2[2]}")
    log(f"  计算得到 R3 矩阵:")
    log(f"  {R3[0]}")
    log(f"  {R3[1]}")
    log(f"  {R3[2]}")

    return [R1, R2, R3], CA


class Protein:
    """
    Usage:
      p = Protein(target_pdb, MatrixDir, TxtSaveDir, structure_id="X", std_pdb_file="std.pdb")
      p.toStructure()
      p.getResidue("output.txt", z_offset_by_resname={"PHE": -11.0})
    """

    def __init__(self, filename, MatrixDir, TxtSaveDir, structure_id, std_pdb_file=None):
        """
        初始化Protein类
        
        参数 Parameters:
        - filename: PDB文件路径 PDB file path
        - MatrixDir: 矩阵文件目录 Matrix file directory
        - TxtSaveDir: 文本保存目录 Text save directory
        - structure_id: 结构ID Structure ID
        - std_pdb_file: 标准PDB文件路径（可选） Standard PDB file path (optional)
        """
        self.filename = filename
        self.structure_id = structure_id
        self.MatrixDir = MatrixDir
        self.TxtSaveDir = TxtSaveDir
        self.std_pdb_file = std_pdb_file
        self.std_vectors = None
        
        # 创建log文件目录
        # Create log file directory
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(TxtSaveDir)), "Log")
        os.makedirs(self.log_dir, exist_ok=True)
        
        import datetime
        # 创建log文件
        # Create log file
        base_name = os.path.splitext(os.path.basename(filename))[0]
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        self.log_file = os.path.join(self.log_dir, f"{base_name}_{date_str}_log.txt")
        
        import datetime
        # 清空log文件
        # Clear log file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"=== 计算日志 ===\n")
            f.write(f"生成时间: {timestamp}\n")
            f.write(f"PDB文件: {filename}\n")
            f.write(f"结构ID: {structure_id}\n")
            f.write(f"标准PDB文件: {std_pdb_file}\n")
            f.write(f"\n")

        if std_pdb_file:
            self.load_std_vectors()
    
    def log_print(self, message):
        """同时打印到控制台和log文件
        
        Print to both console and log file
        """
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{message}\n")

    def load_std_vectors(self):
        """从标准PDB文件中读取标准向量
        
        Read standard vectors from standard PDB file
        """
        parser = PDBParser(PERMISSIVE=1, QUIET=True)
        structure = parser.get_structure("std", self.std_pdb_file)

        model = structure[0]
        self.std_vectors = {}
        first_read = True

        for chain_id, chain in model.chains.items():
            for res in chain.get_list():
                res_name = res.get_resname()
                if res_name not in ("TYR", "PHE"):
                    continue

                res_num = res.id[1]
                atom_dict = {atom.name: atom for atom in res.get_atoms()}

                need = ['CA', 'CB', 'CG', 'CD1', 'N', 'C']
                if not all(a in atom_dict for a in need):
                    continue

                ca = atom_dict['CA'].get_vector()
                cb = atom_dict['CB'].get_vector()
                cg = atom_dict['CG'].get_vector()
                cd1 = atom_dict['CD1'].get_vector()
                n = atom_dict['N'].get_vector()
                c = atom_dict['C'].get_vector()

                # 第一次读取时打印标准原子坐标和名称
                if first_read:
                    self.log_print(f"=== 标准PDB文件原子信息 ===")
                    self.log_print(f"残基: {res_name} {res_num}")
                    for atom_name in need:
                        atom = atom_dict[atom_name]
                        coord = atom.get_vector()
                        self.log_print(f"  {atom_name}: {coord[:3]}")
                    first_read = False

                key_full = (res_name, chain_id, res_num)
                key_no_chain = (res_name, res_num)

                phStd1 = (cg - cb)[:3]
                phStd2 = (cg - cd1)[:3]
                CH2td1 = (cb - ca)[:3]
                CCO_Std = (c - ca)[:3]
                CN_Std = (n - ca)[:3]

                # 打印计算得到的标准向量数值
                self.log_print(f"=== 标准向量计算结果 ===")
                self.log_print(f"残基: {res_name} {res_num}")
                self.log_print(f"  phStd1: {phStd1}")
                self.log_print(f"  phStd2: {phStd2}")
                self.log_print(f"  CH2td1: {CH2td1}")
                self.log_print(f"  CCO_Std: {CCO_Std}")
                self.log_print(f"  CN_Std: {CN_Std}")

                vector_data = {
                    'phStd1': phStd1,
                    'phStd2': phStd2,
                    'CH2td1': CH2td1,
                    'CCO_Std': CCO_Std,
                    'CN_Std': CN_Std,
                    'original_coords': {'CA': ca[:3], 'CD1': cd1[:3]},
                }

                self.std_vectors[key_full] = vector_data
                if key_no_chain not in self.std_vectors:
                    self.std_vectors[key_no_chain] = vector_data

    def toStructure(self):
        """
        将PDB文件解析为结构对象
        
        Parse PDB file into structure object
        """
        parser = PDBParser(PERMISSIVE=1, QUIET=True)
        self.structure = parser.get_structure(self.structure_id, self.filename)

    def getResidue(self, SaveName, z_offset_by_resname=None, z_offset_by_chain=None):
        """
        计算残基的拉曼光谱数据并保存
        
        参数 Parameters:
        - SaveName: 保存文件名 Save filename
        - z_offset_by_resname: 按残基名称的z轴偏移量 Z-axis offset by residue name
        - z_offset_by_chain: 按链的z轴偏移量 Z-axis offset by chain
        
        Compute Raman spectrum data for residues and save
        
        Parameters:
        - SaveName: Save filename
        - z_offset_by_resname: dict, e.g. {"PHE": -11.0} to reproduce previous behavior safely
        - z_offset_by_chain: dict, e.g. {"A": 0.0, "B": -11.0} if needed
        """
        if len(self.structure.models) != 1:
            raise ValueError("当前脚本仅支持单MODEL的PDB。")

        if not self.std_vectors:
            raise ValueError("错误：未加载标准向量，请在初始化时提供标准PDB文件 std_pdb_file。")

        z_offset_by_resname = z_offset_by_resname or {}
        z_offset_by_chain = z_offset_by_chain or {}

        model = self.structure[0]
        DataSet = np.zeros((0, 4), dtype=float)

        for chain_id, chain in model.chains.items():
            aaList = chain.get_list()
            self.log_print(f"链 {chain_id} 中的氨基酸个数：{len(aaList)}")
            self.log_print(f"file={self.filename}")

            # --- Chain-level: build residue_T_dict once ---
            residue_T_dict = {}
            residue_Z_dict = {}
            for res in aaList:
                if res.get_resname() in ("TYR", "PHE"):
                    out = compute_T_mats_for_residue(res, chain_id, self.std_vectors, self.log_print)
                    if out is None:
                        continue
                    mats, CG = out
                    resnum = res.id[1]
                    residue_T_dict[resnum] = mats

                    z = float(CG[2])
                    z += float(z_offset_by_resname.get(res.get_resname(), 0.0))
                    z += float(z_offset_by_chain.get(chain_id, 0.0))
                    residue_Z_dict[resnum] = z

            # --- Per residue: compute Raman dataset ---
            for aa in aaList:
                resname = aa.get_resname()
                if resname not in ("TYR", "PHE"):
                    continue

                resnum = aa.id[1]
                mats = residue_T_dict.get(resnum)
                if mats is None:
                    continue

                # Current residue T_by_label for single-residue modes
                T_by_label = {1: mats[0], 2: mats[1], 3: mats[2]}  # label 4/5 absent -> ignored
                Z = residue_Z_dict.get(resnum, 0.0)

                xlsx_path = os.path.join(self.MatrixDir, f"YYY.xlsx")
                MatrixSet, Label, WaveNumber, Width, residue1, residue2, residue3 = Matrix.LoadMartix(xlsx_path)

                S_Set, WaveSet = Matrix.ProcessMartix(
                    MatrixSet=MatrixSet,
                    WaveNumber=WaveNumber,
                    Label=Label,
                    Z=Z,
                    T_by_label=T_by_label,
                    residue1=residue1, residue2=residue2, residue3=residue3,
                    residue_T_dict=residue_T_dict,
                    chain_id=chain_id,
                    log_print=self.log_print
                )

                temp = Matrix.SaveData(S_Set, WaveSet, Width, residue_id=resnum)
                DataSet = np.vstack([DataSet, temp])

        # remove all-zero rows (safety)
        if DataSet.size == 0:
            raise RuntimeError("未生成任何谱数据：请检查PDB中TYR/PHE原子是否完整，以及Excel文件路径是否正确。")

        import datetime
        DataSet = DataSet[~np.all(np.isclose(DataSet, 0.0), axis=1)]
        # 在文件名中添加处理的文件名和精确到秒的时间
        base_name = os.path.splitext(SaveName)[0]
        # 提取处理的文件名（不包含路径和扩展名）
        processed_filename = os.path.splitext(os.path.basename(self.filename))[0]
        # 生成包含日期和时间（精确到秒）的字符串
        datetime_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_name_with_datetime = f"Raman_{processed_filename}_{datetime_str}.txt"
        save_path = os.path.join(self.TxtSaveDir, save_name_with_datetime)
        SaveFiles(DataSet=DataSet, SavePath=save_path, input_file=self.filename)
        self.log_print(f"完成：已保存到 {save_path}")
        self.log_print(f"计算日志已保存到：{self.log_file}")


# =========================================================
#  Example usage (edit paths, then run)
# =========================================================
if __name__ == "__main__":
    # ---- 用户需要修改的路径 ----
    target_pdb = r"test.pdb"
    std_pdb = r"YYY.pdb"
    matrix_dir = r"./MatrixDir"      # 存放 TYR.xlsx / PHE.xlsx
    out_dir = r"./Output"
    out_name = "raman.txt"

    # 若你需要复现原来 PHE Z-11 的行为，请保留下面这一行
    z_offset_by_resname = {"PHE": 0.0}

    p = Protein(
        filename=target_pdb,
        MatrixDir=matrix_dir,
        TxtSaveDir=out_dir,
        structure_id="protein",
        std_pdb_file=std_pdb
    )
    p.toStructure()
    p.getResidue(out_name, z_offset_by_resname=z_offset_by_resname)

