/**
 * 三层架构Context统一导出
 *
 * 架构层次：
 * - ProviderContext: 底层 - 管理AI服务商配置（API key、host等）
 * - ModelContext: 中层 - 管理模型列表（系统预设+用户collection）
 * - DefaultsContext: 顶层 - 管理默认模型选择
 */

// 三层架构
export { ProviderProvider, useProvider } from './ProviderContext'
export { ModelProvider, useModel } from './ModelContext'
export { DefaultsProvider, useDefaults } from './DefaultsContext'

/**
 * 三层架构组合Provider
 * 按顺序包装：Provider -> Model -> Defaults
 *
 * 使用示例：
 * <ThreeLayerProvider>
 *   <App />
 * </ThreeLayerProvider>
 */
import React, { ReactNode } from 'react'
import { ProviderProvider } from './ProviderContext'
import { ModelProvider } from './ModelContext'
import { DefaultsProvider } from './DefaultsContext'

export function ThreeLayerProvider({ children }: { children: ReactNode }) {
    return (
        <ProviderProvider>
            <ModelProvider>
                <DefaultsProvider>
                    {children}
                </DefaultsProvider>
            </ModelProvider>
        </ProviderProvider>
    )
}
