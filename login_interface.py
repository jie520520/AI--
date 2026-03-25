"""
登录界面集成模块 - Login Interface Integration

在主应用中集成登录功能
"""

import streamlit as st
from user_auth import UserManager


def show_login_page():
    """显示登录页面"""
    st.markdown("""
    <div style="text-align: center; padding: 50px 0 30px 0;">
        <h1>🔐 AI彩票量化研究系统</h1>
        <h3>内部测试版 - 用户登录</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # 居中显示登录表单
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 请登录")
        
        # 登录表单
        username = st.text_input(
            "用户名",
            placeholder="请输入用户名",
            key="login_username"
        )
        
        password = st.text_input(
            "密码",
            type="password",
            placeholder="请输入密码",
            key="login_password"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔓 登录", use_container_width=True, type="primary"):
                if not username or not password:
                    st.error("请输入用户名和密码")
                else:
                    # 验证用户
                    manager = UserManager()
                    success, message = manager.verify_user(username, password)
                    
                    if success:
                        # 登录成功
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.is_admin = manager.is_admin(username)
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
        
        with col_btn2:
            if st.button("❓ 忘记密码", use_container_width=True):
                st.info("请联系管理员重置密码")
        
        st.divider()
        
        # 说明信息
        st.info("""
        **系统说明：**
        
        ⚠️ 本系统仅限内部测试使用
        
        - 不开放注册
        - 仅限授权用户登录
        - 如需账号请联系管理员
        
        **默认管理员账号（首次安装）：**
        - 用户名：admin
        - 密码：admin123
        - ⚠️ 请立即登录并修改密码！
        """)


def show_user_management():
    """显示用户管理界面（仅管理员）"""
    if not st.session_state.get('is_admin', False):
        st.error("❌ 仅限管理员访问")
        return
    
    st.markdown("### 👥 用户管理")
    
    manager = UserManager()
    
    # 统计信息
    stats = manager.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总用户数", stats['total_users'])
    with col2:
        st.metric("活跃用户", stats['active_users'])
    with col3:
        st.metric("禁用用户", stats['inactive_users'])
    with col4:
        st.metric("管理员", stats['admin_users'])
    
    st.divider()
    
    # 用户管理选项卡
    tab1, tab2, tab3 = st.tabs(["➕ 添加用户", "📋 用户列表", "🔧 其他操作"])
    
    with tab1:
        st.markdown("#### 添加新用户")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_username = st.text_input(
                "用户名",
                placeholder="至少3个字符",
                key="new_username"
            )
            
            new_role = st.selectbox(
                "角色",
                options=["user", "admin"],
                format_func=lambda x: "管理员" if x == "admin" else "普通用户",
                key="new_role"
            )
        
        with col2:
            new_password = st.text_input(
                "密码",
                type="password",
                placeholder="至少6个字符",
                key="new_password"
            )
            
            new_note = st.text_input(
                "备注（可选）",
                placeholder="用户说明",
                key="new_note"
            )
        
        if st.button("➕ 添加用户", type="primary", use_container_width=True):
            if not new_username or not new_password:
                st.error("用户名和密码不能为空")
            else:
                success, message = manager.add_user(
                    new_username, 
                    new_password, 
                    new_role, 
                    new_note
                )
                
                if success:
                    st.success(message)
                    # 清空输入
                    st.rerun()
                else:
                    st.error(message)
    
    with tab2:
        st.markdown("#### 用户列表")
        
        users = manager.list_users()
        
        if users:
            for user in users:
                with st.expander(
                    f"{'🔴' if not user['is_active'] else '🟢'} {user['username']} "
                    f"({'管理员' if user['role'] == 'admin' else '普通用户'})"
                ):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**创建时间：** {user['created_at'][:10]}")
                        st.write(f"**最后登录：** {user['last_login'] if user['last_login'] and user['last_login'] != '未登录' else '未登录'}")
                        st.write(f"**状态：** {'启用' if user['is_active'] else '禁用'}")
                    
                    with col2:
                        st.write(f"**角色：** {'管理员' if user['role'] == 'admin' else '普通用户'}")
                        st.write(f"**备注：** {user['note'] if user['note'] else '无'}")
                    
                    # 操作按钮
                    col_btn1, col_btn2, col_btn3 = st.columns(3)
                    
                    with col_btn1:
                        if user['username'] != 'admin':
                            if st.button(
                                "🔄 切换状态", 
                                key=f"toggle_{user['username']}",
                                use_container_width=True
                            ):
                                success, message = manager.toggle_user_status(user['username'])
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                    
                    with col_btn2:
                        if st.button(
                            "🔑 重置密码", 
                            key=f"reset_{user['username']}",
                            use_container_width=True
                        ):
                            st.session_state[f'reset_password_{user["username"]}'] = True
                    
                    with col_btn3:
                        if user['username'] not in ['admin', st.session_state.username]:
                            if st.button(
                                "🗑️ 删除", 
                                key=f"delete_{user['username']}",
                                use_container_width=True
                            ):
                                success, message = manager.delete_user(
                                    user['username'], 
                                    st.session_state.username
                                )
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                    
                    # 重置密码对话框
                    if st.session_state.get(f'reset_password_{user["username"]}', False):
                        new_pwd = st.text_input(
                            "新密码",
                            type="password",
                            key=f"new_pwd_{user['username']}"
                        )
                        
                        col_confirm1, col_confirm2 = st.columns(2)
                        with col_confirm1:
                            if st.button(
                                "✓ 确认", 
                                key=f"confirm_reset_{user['username']}",
                                use_container_width=True
                            ):
                                if new_pwd:
                                    success, message = manager.reset_password(
                                        user['username'],
                                        new_pwd,
                                        st.session_state.username
                                    )
                                    if success:
                                        st.success(message)
                                        st.session_state[f'reset_password_{user["username"]}'] = False
                                        st.rerun()
                                    else:
                                        st.error(message)
                                else:
                                    st.error("请输入新密码")
                        
                        with col_confirm2:
                            if st.button(
                                "✗ 取消", 
                                key=f"cancel_reset_{user['username']}",
                                use_container_width=True
                            ):
                                st.session_state[f'reset_password_{user["username"]}'] = False
                                st.rerun()
        else:
            st.info("暂无用户")
    
    with tab3:
        st.markdown("#### 修改密码")
        
        col1, col2 = st.columns(2)
        
        with col1:
            old_pwd = st.text_input(
                "旧密码",
                type="password",
                key="change_old_pwd"
            )
        
        with col2:
            new_pwd = st.text_input(
                "新密码",
                type="password",
                key="change_new_pwd"
            )
        
        if st.button("🔑 修改我的密码", type="primary", use_container_width=True):
            if not old_pwd or not new_pwd:
                st.error("请输入旧密码和新密码")
            else:
                success, message = manager.change_password(
                    st.session_state.username,
                    old_pwd,
                    new_pwd
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)


def show_logout_button():
    """显示登出按钮"""
    if st.session_state.get('logged_in', False):
        # 在侧边栏显示用户信息和登出按钮
        with st.sidebar:
            st.divider()
            st.markdown(f"**👤 当前用户：** {st.session_state.username}")
            if st.session_state.get('is_admin', False):
                st.markdown("**🔑 角色：** 管理员")
            
            if st.button("🚪 登出", use_container_width=True):
                # 清除登录状态
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.is_admin = False
                st.rerun()


def require_login():
    """
    要求登录（在主应用开头调用）
    
    Returns:
        bool: 是否已登录
    """
    # 初始化登录状态
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.is_admin = False
    
    # 如果未登录，显示登录页面
    if not st.session_state.logged_in:
        show_login_page()
        st.stop()
    
    # 已登录，显示登出按钮
    show_logout_button()
    
    return True


# 快捷函数
def is_logged_in() -> bool:
    """检查是否已登录"""
    return st.session_state.get('logged_in', False)


def get_current_user() -> str:
    """获取当前用户名"""
    return st.session_state.get('username', '')


def is_admin() -> bool:
    """检查当前用户是否是管理员"""
    return st.session_state.get('is_admin', False)
