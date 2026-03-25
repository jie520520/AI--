"""
用户认证管理系统 - User Authentication System

特点：
1. 仅限内部测试使用
2. 不开放注册
3. 只有后台注册的用户才能登录
4. 简单安全的认证机制
"""

import json
import hashlib
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class UserManager:
    """用户管理器"""
    
    def __init__(self, users_file: str = "users/users.json"):
        """
        初始化用户管理器
        
        Args:
            users_file: 用户数据文件路径
        """
        self.users_file = users_file
        self.users_dir = os.path.dirname(users_file)
        
        # 确保目录存在
        if self.users_dir and not os.path.exists(self.users_dir):
            os.makedirs(self.users_dir)
        
        # 初始化用户数据
        self._initialize_users()
    
    def _initialize_users(self):
        """初始化用户数据（首次运行时创建默认管理员）"""
        if not os.path.exists(self.users_file):
            # 创建默认管理员账号
            default_admin = {
                "users": {
                    "admin": {
                        "password_hash": self._hash_password("admin123"),
                        "role": "admin",
                        "created_at": datetime.now().isoformat(),
                        "last_login": None,
                        "is_active": True,
                        "note": "默认管理员账号"
                    }
                },
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(default_admin, f, ensure_ascii=False, indent=2)
    
    def _hash_password(self, password: str) -> str:
        """
        密码哈希
        
        Args:
            password: 明文密码
            
        Returns:
            哈希后的密码
        """
        # 使用SHA256哈希
        return hashlib.sha256(password.encode('utf-8')).hexdigest()
    
    def _load_users(self) -> Dict:
        """加载用户数据"""
        try:
            with open(self.users_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {"users": {}, "created_at": datetime.now().isoformat(), "version": "1.0"}
    
    def _save_users(self, data: Dict):
        """保存用户数据"""
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def verify_user(self, username: str, password: str) -> Tuple[bool, str]:
        """
        验证用户
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            (是否验证成功, 消息)
        """
        data = self._load_users()
        users = data.get('users', {})
        
        # 检查用户是否存在
        if username not in users:
            return False, "用户名不存在"
        
        user = users[username]
        
        # 检查用户是否被禁用
        if not user.get('is_active', True):
            return False, "账号已被禁用"
        
        # 验证密码
        password_hash = self._hash_password(password)
        if password_hash != user['password_hash']:
            return False, "密码错误"
        
        # 更新最后登录时间
        user['last_login'] = datetime.now().isoformat()
        data['users'][username] = user
        self._save_users(data)
        
        return True, "登录成功"
    
    def add_user(
        self, 
        username: str, 
        password: str, 
        role: str = "user",
        note: str = ""
    ) -> Tuple[bool, str]:
        """
        添加用户（仅限后台操作）
        
        Args:
            username: 用户名
            password: 密码
            role: 角色（admin/user）
            note: 备注
            
        Returns:
            (是否成功, 消息)
        """
        data = self._load_users()
        users = data.get('users', {})
        
        # 检查用户名是否已存在
        if username in users:
            return False, "用户名已存在"
        
        # 检查用户名格式
        if not username or len(username) < 3:
            return False, "用户名至少3个字符"
        
        # 检查密码强度
        if not password or len(password) < 6:
            return False, "密码至少6个字符"
        
        # 添加用户
        users[username] = {
            "password_hash": self._hash_password(password),
            "role": role,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "is_active": True,
            "note": note
        }
        
        data['users'] = users
        self._save_users(data)
        
        return True, f"用户 {username} 添加成功"
    
    def delete_user(self, username: str, admin_username: str) -> Tuple[bool, str]:
        """
        删除用户（仅限管理员）
        
        Args:
            username: 要删除的用户名
            admin_username: 执行删除的管理员用户名
            
        Returns:
            (是否成功, 消息)
        """
        data = self._load_users()
        users = data.get('users', {})
        
        # 检查要删除的用户是否存在
        if username not in users:
            return False, "用户不存在"
        
        # 不能删除自己
        if username == admin_username:
            return False, "不能删除自己"
        
        # 不能删除默认管理员
        if username == "admin":
            return False, "不能删除默认管理员"
        
        # 删除用户
        del users[username]
        data['users'] = users
        self._save_users(data)
        
        return True, f"用户 {username} 已删除"
    
    def toggle_user_status(self, username: str) -> Tuple[bool, str]:
        """
        切换用户状态（启用/禁用）
        
        Args:
            username: 用户名
            
        Returns:
            (是否成功, 消息)
        """
        data = self._load_users()
        users = data.get('users', {})
        
        if username not in users:
            return False, "用户不存在"
        
        # 不能禁用默认管理员
        if username == "admin":
            return False, "不能禁用默认管理员"
        
        # 切换状态
        current_status = users[username].get('is_active', True)
        users[username]['is_active'] = not current_status
        
        data['users'] = users
        self._save_users(data)
        
        new_status = "启用" if not current_status else "禁用"
        return True, f"用户 {username} 已{new_status}"
    
    def change_password(
        self, 
        username: str, 
        old_password: str, 
        new_password: str
    ) -> Tuple[bool, str]:
        """
        修改密码
        
        Args:
            username: 用户名
            old_password: 旧密码
            new_password: 新密码
            
        Returns:
            (是否成功, 消息)
        """
        data = self._load_users()
        users = data.get('users', {})
        
        if username not in users:
            return False, "用户不存在"
        
        # 验证旧密码
        old_hash = self._hash_password(old_password)
        if old_hash != users[username]['password_hash']:
            return False, "旧密码错误"
        
        # 检查新密码强度
        if not new_password or len(new_password) < 6:
            return False, "新密码至少6个字符"
        
        # 更新密码
        users[username]['password_hash'] = self._hash_password(new_password)
        data['users'] = users
        self._save_users(data)
        
        return True, "密码修改成功"
    
    def reset_password(
        self, 
        username: str, 
        new_password: str, 
        admin_username: str
    ) -> Tuple[bool, str]:
        """
        重置密码（仅限管理员）
        
        Args:
            username: 要重置的用户名
            new_password: 新密码
            admin_username: 执行重置的管理员
            
        Returns:
            (是否成功, 消息)
        """
        data = self._load_users()
        users = data.get('users', {})
        
        if username not in users:
            return False, "用户不存在"
        
        # 检查管理员权限
        if admin_username not in users:
            return False, "管理员不存在"
        
        if users[admin_username].get('role') != 'admin':
            return False, "无管理员权限"
        
        # 检查新密码强度
        if not new_password or len(new_password) < 6:
            return False, "新密码至少6个字符"
        
        # 重置密码
        users[username]['password_hash'] = self._hash_password(new_password)
        data['users'] = users
        self._save_users(data)
        
        return True, f"用户 {username} 密码已重置"
    
    def list_users(self) -> List[Dict]:
        """
        列出所有用户
        
        Returns:
            用户列表
        """
        data = self._load_users()
        users = data.get('users', {})
        
        user_list = []
        for username, user_data in users.items():
            user_list.append({
                'username': username,
                'role': user_data.get('role', 'user'),
                'created_at': user_data.get('created_at', ''),
                'last_login': user_data.get('last_login', '未登录'),
                'is_active': user_data.get('is_active', True),
                'note': user_data.get('note', '')
            })
        
        # 按创建时间排序
        user_list.sort(key=lambda x: x['created_at'])
        
        return user_list
    
    def get_user_info(self, username: str) -> Optional[Dict]:
        """
        获取用户信息
        
        Args:
            username: 用户名
            
        Returns:
            用户信息（不包含密码）
        """
        data = self._load_users()
        users = data.get('users', {})
        
        if username not in users:
            return None
        
        user_data = users[username].copy()
        # 不返回密码哈希
        user_data.pop('password_hash', None)
        user_data['username'] = username
        
        return user_data
    
    def is_admin(self, username: str) -> bool:
        """
        检查是否是管理员
        
        Args:
            username: 用户名
            
        Returns:
            是否是管理员
        """
        data = self._load_users()
        users = data.get('users', {})
        
        if username not in users:
            return False
        
        return users[username].get('role') == 'admin'
    
    def get_statistics(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            统计数据
        """
        data = self._load_users()
        users = data.get('users', {})
        
        total = len(users)
        active = len([u for u in users.values() if u.get('is_active', True)])
        admins = len([u for u in users.values() if u.get('role') == 'admin'])
        
        # 最近登录
        recent_logins = []
        for username, user_data in users.items():
            if user_data.get('last_login'):
                recent_logins.append({
                    'username': username,
                    'last_login': user_data['last_login']
                })
        
        recent_logins.sort(key=lambda x: x['last_login'], reverse=True)
        
        return {
            'total_users': total,
            'active_users': active,
            'inactive_users': total - active,
            'admin_users': admins,
            'recent_logins': recent_logins[:5]
        }


def create_default_users():
    """创建默认用户（首次安装时使用）"""
    manager = UserManager()
    
    # 默认管理员已在初始化时创建
    # 可以在这里添加其他默认用户
    
    print("默认用户创建完成")
    print("默认管理员账号：admin")
    print("默认密码：admin123")
    print("⚠️ 请立即登录并修改默认密码！")


if __name__ == "__main__":
    # 测试
    create_default_users()
    
    manager = UserManager()
    
    # 测试添加用户
    success, msg = manager.add_user("test_user", "test123", note="测试用户")
    print(f"添加用户: {msg}")
    
    # 测试验证
    success, msg = manager.verify_user("admin", "admin123")
    print(f"验证管理员: {msg}")
    
    # 测试列出用户
    users = manager.list_users()
    print(f"用户列表: {len(users)}个用户")
    
    # 测试统计
    stats = manager.get_statistics()
    print(f"统计: {stats}")
